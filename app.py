from __future__ import annotations

import io
from pathlib import Path
from urllib.parse import quote

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import torch
import torch.nn.functional as F
from fpdf import FPDF
from rdkit import Chem
# from rdkit.Chem import Descriptors, Draw, rdMolDescriptors
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.rdchem import Atom, HybridizationType
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool


# ============================================================
# APP CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AqSolDB Solubility Predictor",
    page_icon="🧪",
    layout="wide",
)

PROJECT_ROOT = Path(__file__).resolve().parent

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "gnn_experiments"
    / "gcn_scaffold"
    / "best_model.pt"
)

TEST_CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "aqsoldb_scaffold_test.csv"
)

PREDICTIONS_CSV_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "gnn_experiments"
    / "gcn_scaffold"
    / "test_predictions.csv"
)

TEST_R2 = 0.7861
TEST_RMSE = 1.0946
TEST_MAE = 0.7506


# ============================================================
# TRAINING-MATCHED GRAPH FEATURES
# ============================================================

ATOM_SYMBOLS = [
    "C",
    "N",
    "O",
    "S",
    "P",
    "F",
    "Cl",
    "Br",
    "I",
    "B",
    "Si",
]

DEGREE_VALUES = [0, 1, 2, 3, 4, 5]
FORMAL_CHARGE_VALUES = [-2, -1, 0, 1, 2]

HYBRIDIZATION_VALUES = [
    HybridizationType.SP,
    HybridizationType.SP2,
    HybridizationType.SP3,
    HybridizationType.SP3D,
    HybridizationType.SP3D2,
]

TOTAL_H_VALUES = [0, 1, 2, 3, 4]


def one_hot_with_unknown(value, allowed_values):
    encoding = [0.0] * (len(allowed_values) + 1)

    try:
        index = allowed_values.index(value)
    except ValueError:
        index = len(allowed_values)

    encoding[index] = 1.0
    return encoding


def atom_to_features(atom: Atom):
    features = []

    features.extend(
        one_hot_with_unknown(
            atom.GetSymbol(),
            ATOM_SYMBOLS,
        )
    )

    features.extend(
        one_hot_with_unknown(
            atom.GetDegree(),
            DEGREE_VALUES,
        )
    )

    features.extend(
        one_hot_with_unknown(
            atom.GetFormalCharge(),
            FORMAL_CHARGE_VALUES,
        )
    )

    features.extend(
        one_hot_with_unknown(
            atom.GetHybridization(),
            HYBRIDIZATION_VALUES,
        )
    )

    features.extend(
        one_hot_with_unknown(
            atom.GetTotalNumHs(),
            TOTAL_H_VALUES,
        )
    )

    features.append(
        float(atom.GetIsAromatic())
    )

    features.append(
        float(atom.IsInRing())
    )

    features.append(
        atom.GetMass() / 100.0
    )

    return features


# ============================================================
# MODEL
# ============================================================

class GCNRegressor(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers,
        dropout,
    ):
        super().__init__()

        self.dropout = dropout
        self.convolutions = nn.ModuleList()
        self.normalizations = nn.ModuleList()

        self.convolutions.append(
            GCNConv(
                input_dim,
                hidden_dim,
            )
        )

        self.normalizations.append(
            nn.BatchNorm1d(
                hidden_dim
            )
        )

        for _ in range(
            num_layers - 1
        ):
            self.convolutions.append(
                GCNConv(
                    hidden_dim,
                    hidden_dim,
                )
            )

            self.normalizations.append(
                nn.BatchNorm1d(
                    hidden_dim
                )
            )

        self.regression_head = nn.Sequential(
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                hidden_dim,
                1,
            ),
        )

    def forward(
        self,
        data,
    ):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        for convolution, normalization in zip(
            self.convolutions,
            self.normalizations,
        ):
            x = convolution(
                x,
                edge_index,
            )

            x = normalization(
                x
            )

            x = F.relu(
                x
            )

            x = F.dropout(
                x,
                p=self.dropout,
                training=self.training,
            )

        graph_embedding = global_mean_pool(
            x,
            batch,
        )

        prediction = self.regression_head(
            graph_embedding
        )

        return prediction.view(-1)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    model = GCNRegressor(
        input_dim=int(
            checkpoint["node_feature_dim"]
        ),
        hidden_dim=int(
            checkpoint["hidden_dim"]
        ),
        num_layers=int(
            checkpoint["num_layers"]
        ),
        dropout=float(
            checkpoint["dropout"]
        ),
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model, checkpoint


# ============================================================
# INPUT + GRAPH CONVERSION
# ============================================================

def is_smiles(
    value: str,
) -> bool:
    value = value.strip()

    if not value:
        return False

    return (
        Chem.MolFromSmiles(
            value
        )
        is not None
    )


@st.cache_data(
    show_spinner=False
)
def name_to_smiles(
    name: str,
):
    safe_name = quote(
        name.strip(),
        safe="",
    )

    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/"
        f"compound/name/{safe_name}/property/"
        "CanonicalSMILES/TXT"
    )

    try:
        response = requests.get(
            url,
            timeout=10,
        )

    except requests.RequestException:
        return None

    if response.status_code != 200:
        return None

    smiles = response.text.strip()

    return (
        smiles
        or None
    )


def smiles_to_graph(
    smiles: str,
):
    molecule = Chem.MolFromSmiles(
        smiles
    )

    if molecule is None:
        raise ValueError(
            "Invalid SMILES."
        )

    x = torch.tensor(
        [
            atom_to_features(atom)
            for atom
            in molecule.GetAtoms()
        ],
        dtype=torch.float32,
    )

    source_nodes = []
    destination_nodes = []

    for bond in molecule.GetBonds():
        begin_index = (
            bond.GetBeginAtomIdx()
        )

        end_index = (
            bond.GetEndAtomIdx()
        )

        source_nodes.extend(
            [
                begin_index,
                end_index,
            ]
        )

        destination_nodes.extend(
            [
                end_index,
                begin_index,
            ]
        )

    if source_nodes:
        edge_index = torch.tensor(
            [
                source_nodes,
                destination_nodes,
            ],
            dtype=torch.long,
        )

    else:
        edge_index = torch.empty(
            (
                2,
                0,
            ),
            dtype=torch.long,
        )

    return Data(
        x=x,
        edge_index=edge_index,
    )


def add_batch_vector(
    graph: Data,
) -> Data:
    graph.batch = torch.zeros(
        graph.x.shape[0],
        dtype=torch.long,
    )

    return graph


@torch.no_grad()
def predict_logS(
    model,
    checkpoint,
    smiles,
):
    graph = add_batch_vector(
        smiles_to_graph(
            smiles
        )
    )

    normalized_prediction = float(
        model(
            graph
        ).item()
    )

    target_mean = float(
        checkpoint["target_mean"]
    )

    target_std = float(
        checkpoint["target_std"]
    )

    return (
        normalized_prediction
        * target_std
        + target_mean
    )


# ============================================================
# REAL TEST-SET PREDICTIONS FOR THE PLOT
# ============================================================

@st.cache_data(
    show_spinner=False
)
def load_or_compute_test_predictions(
    _model,
    target_mean,
    target_std,
):
    if PREDICTIONS_CSV_PATH.exists():
        prediction_df = pd.read_csv(
            PREDICTIONS_CSV_PATH
        )

        if "predicted_logS" in prediction_df.columns:
            return (
                prediction_df[
                    "predicted_logS"
                ]
                .astype(float)
                .to_numpy()
            )

    if not TEST_CSV_PATH.exists():
        return np.array(
            [],
            dtype=float,
        )

    test_df = pd.read_csv(
        TEST_CSV_PATH
    )

    graphs = [
        smiles_to_graph(
            str(smiles)
        )
        for smiles
        in test_df["smiles"]
    ]

    loader = DataLoader(
        graphs,
        batch_size=64,
        shuffle=False,
        num_workers=0,
    )

    values = []

    with torch.no_grad():
        for batch in loader:
            normalized = (
                _model(
                    batch
                )
                .detach()
                .cpu()
                .numpy()
            )

            original_scale = (
                normalized
                * target_std
                + target_mean
            )

            values.extend(
                original_scale.tolist()
            )

    return np.asarray(
        values,
        dtype=float,
    )


# ============================================================
# DESCRIPTORS
# ============================================================

def calculate_descriptors(
    molecule,
):
    return {
        "Molecular weight": float(
            Descriptors.MolWt(
                molecule
            )
        ),
        "LogP": float(
            Descriptors.MolLogP(
                molecule
            )
        ),
        "TPSA": float(
            Descriptors.TPSA(
                molecule
            )
        ),
        "H-bond donors": float(
            rdMolDescriptors.CalcNumHBD(
                molecule
            )
        ),
        "H-bond acceptors": float(
            rdMolDescriptors.CalcNumHBA(
                molecule
            )
        ),
        "Rotatable bonds": float(
            Descriptors.NumRotatableBonds(
                molecule
            )
        ),
    }


def solubility_label(
    log_s,
):
    if log_s >= -1:
        return "High"

    if log_s >= -3:
        return "Moderate"

    return "Low"


# ============================================================
# PDF REPORT
# ============================================================

def generate_pdf(
    query,
    smiles,
    log_s,
    descriptors,
):
    pdf = FPDF()

    pdf.add_page()

    pdf.set_font(
        "Arial",
        "B",
        16,
    )

    pdf.cell(
        0,
        10,
        "AqSolDB Solubility Prediction",
        ln=True,
        align="C",
    )

    pdf.ln(
        4
    )

    pdf.set_font(
        "Arial",
        "",
        11,
    )

    lines = [
        f"Input: {query}",
        f"SMILES: {smiles}",
        f"Predicted logS: {log_s:.4f}",
        (
            "Model-derived approximate molar solubility: "
            f"{10 ** log_s:.4e} mol/L"
        ),
        (
            "Qualitative category: "
            f"{solubility_label(log_s)}"
        ),
        "",
        "Molecular descriptors:",
    ]

    for line in lines:
        pdf.set_x(
            pdf.l_margin
        )

        pdf.multi_cell(
            0,
            7,
            line,
        )

    for label, value in descriptors.items():
        pdf.set_x(
            pdf.l_margin
        )

        pdf.multi_cell(
            0,
            7,
            f"- {label}: {value:.3f}",
        )

    pdf.ln(
        3
    )

    pdf.set_x(
        pdf.l_margin
    )

    pdf.multi_cell(
        0,
        7,
        (
            "Model: GCN regression model trained on AqSolDB "
            "with scaffold split. "
            f"Test R2={TEST_R2:.4f}, "
            f"RMSE={TEST_RMSE:.4f}, "
            f"MAE={TEST_MAE:.4f}."
        ),
    )

    output = pdf.output(
        dest="S"
    )

    if isinstance(
        output,
        str,
    ):
        return output.encode(
            "latin-1"
        )

    return bytes(
        output
    )


# ============================================================
# UI
# ============================================================

st.title(
    "🧪 Solubility Predictor (AqSolDB)"
)

st.write(
    "Enter a molecule name or SMILES to predict aqueous solubility "
    "using the trained GCN scaffold-split model."
)


try:
    model, checkpoint = load_model()

except Exception as error:
    st.error(
        str(error)
    )
    st.stop()


with st.sidebar:
    st.header(
        "ℹ️ About This App"
    )

    st.write(
        "This app predicts aqueous solubility (logS) "
        "of small molecules using a Graph Neural Network."
    )

    st.markdown(
        f"""
**Model:** GCN  
**Dataset:** AqSolDB  
**Split:** Scaffold  
**Best epoch:** {checkpoint["epoch"]}  
**Node features:** {checkpoint["node_feature_dim"]}  
**Hidden dimension:** {checkpoint["hidden_dim"]}  
**Layers:** {checkpoint["num_layers"]}  
**Dropout:** {checkpoint["dropout"]}
"""
    )

    st.divider()

    st.subheader(
        "Model Performance"
    )

    st.metric(
        "R²",
        f"{TEST_R2:.4f}",
    )

    st.metric(
        "RMSE",
        f"{TEST_RMSE:.4f}",
    )

    st.metric(
        "MAE",
        f"{TEST_MAE:.4f}",
    )


st.subheader(
    "📊 Model Performance"
)

metric_col1, metric_col2, metric_col3 = st.columns(
    3
)

with metric_col1:
    st.metric(
        "R² Score",
        f"{TEST_R2:.4f}",
    )

with metric_col2:
    st.metric(
        "RMSE",
        f"{TEST_RMSE:.4f}",
    )

with metric_col3:
    st.metric(
        "MAE",
        f"{TEST_MAE:.4f}",
    )


query = st.text_input(
    "🔍 Enter Molecule Name or SMILES",
    value="CCO",
    placeholder="Examples: ethanol, caffeine, CCO",
)


if st.button(
    "✨ Predict Solubility",
    type="primary",
):
    clean_query = query.strip()

    if not clean_query:
        st.warning(
            "Enter a molecule name or SMILES."
        )
        st.stop()

    if is_smiles(
        clean_query
    ):
        smiles = clean_query
        source = "SMILES input"

    else:
        with st.spinner(
            "Looking up molecule in PubChem..."
        ):
            smiles = name_to_smiles(
                clean_query
            )

        source = "PubChem name lookup"

    if not smiles:
        st.error(
            "Could not resolve this molecule. "
            "Try a valid SMILES."
        )
        st.stop()

    molecule = Chem.MolFromSmiles(
        smiles
    )

    if molecule is None:
        st.error(
            "Resolved structure is not a valid SMILES."
        )
        st.stop()

    try:
        predicted_log_s = predict_logS(
            model,
            checkpoint,
            smiles,
        )

    except Exception as error:
        st.error(
            f"Prediction failed: {error}"
        )
        st.stop()

    descriptors = calculate_descriptors(
        molecule
    )

    molar_solubility = (
        10
        ** predicted_log_s
    )

    category = solubility_label(
        predicted_log_s
    )

    st.divider()

    molecule_col, prediction_col = st.columns(
        [
            1,
            1.2,
        ],
        gap="large",
    )

    with molecule_col:
        st.subheader(
            "Molecule"
        )

        molecule_image = Draw.MolToImage(
            molecule,
            size=(
                420,
                300,
            ),
        )

        st.image(
            molecule_image,
            caption=smiles,
            use_container_width=True,
        )

        st.caption(
            f"Resolved from: {source}"
        )

    with prediction_col:
        st.subheader(
            "Prediction"
        )

        p1, p2 = st.columns(
            2
        )

        with p1:
            st.metric(
                "Predicted logS",
                f"{predicted_log_s:.4f}",
            )

        with p2:
            st.metric(
                "Model-derived approximate molar solubility",
                f"{molar_solubility:.3e} M",
            )

        st.info(
            f"Qualitative solubility: **{category}**"
        )

        st.caption(
            "The molar value is obtained from 10^logS and should be "
            "treated as a model-derived approximation."
        )

    st.divider()

    st.subheader(
        "📈 Prediction vs. Test-Set Distribution"
    )

    target_mean = float(
        checkpoint["target_mean"]
    )

    target_std = float(
        checkpoint["target_std"]
    )

    with st.spinner(
        "Loading real test-set predictions..."
    ):
        test_predictions = (
            load_or_compute_test_predictions(
                model,
                target_mean,
                target_std,
            )
        )

    if test_predictions.size > 0:
        figure, axis = plt.subplots(
            figsize=(
                9,
                4.5,
            )
        )

        sample_index = np.arange(
            test_predictions.size
        )

        axis.scatter(
            sample_index,
            test_predictions,
            alpha=0.45,
            s=22,
            label="GCN test-set predictions",
        )

        axis.axhline(
            predicted_log_s,
            linestyle="--",
            linewidth=2,
            label=(
                f"{clean_query}: "
                f"{predicted_log_s:.2f}"
            ),
        )

        axis.set_xlabel(
            "Test molecule index"
        )

        axis.set_ylabel(
            "Predicted logS"
        )

        axis.set_title(
            "Current molecule relative to real scaffold-test predictions"
        )

        axis.legend()

        figure.tight_layout()

        st.pyplot(
            figure
        )

        plt.close(
            figure
        )

    else:
        st.info(
            "Test-set prediction data were not available for the plot."
        )

    st.divider()

    st.subheader(
        "Molecular Descriptors"
    )

    descriptor_columns = st.columns(
        3
    )

    for index, (
        label,
        value,
    ) in enumerate(
        descriptors.items()
    ):
        with descriptor_columns[
            index % 3
        ]:
            st.metric(
                label,
                f"{value:.3f}",
            )

    st.divider()

    st.success(
        (
            f"💧 Predicted logS for "
            f"{clean_query}: "
            f"{predicted_log_s:.4f}"
        )
    )

    pdf_bytes = generate_pdf(
        clean_query,
        smiles,
        predicted_log_s,
        descriptors,
    )

    safe_name = "".join(
        character
        if character.isalnum()
        else "_"
        for character
        in clean_query
    ).strip(
        "_"
    ) or "molecule"

    st.download_button(
        "📄 Download PDF Report",
        data=pdf_bytes,
        file_name=(
            f"{safe_name}_solubility_report.pdf"
        ),
        mime="application/pdf",
    )

    with st.expander(
        "Technical details"
    ):
        st.write(
            f"Checkpoint: `{CHECKPOINT_PATH}`"
        )

        st.write(
            (
                "Target mean: "
                f"{checkpoint['target_mean']:.6f}"
            )
        )

        st.write(
            (
                "Target std: "
                f"{checkpoint['target_std']:.6f}"
            )
        )

        st.write(
            "The prediction is inverse-transformed from the "
            "training-normalized target back to the original logS scale."
        )
