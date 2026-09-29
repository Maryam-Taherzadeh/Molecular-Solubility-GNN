# ============================================================
# PREDICT SOLUBILITY (logS) WITH A SAVED GCN/GAT CHECKPOINT
# ============================================================

from pathlib import Path
import argparse

import torch
from torch_geometric.data import Batch

from train_gnn import (
    create_model,
    smiles_to_graph,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "gnn_experiments"
    / "gcn_scaffold"
    / "best_model.pt"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict aqueous solubility (logS) from a SMILES string."
    )

    parser.add_argument(
        "--smiles",
        type=str,
        required=True,
        help="Molecular SMILES string.",
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Path to the saved model checkpoint.",
    )

    return parser.parse_args()


def choose_device(model_name: str) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    # GAT scatter operations may not be fully supported on Apple MPS.
    if model_name == "gat":
        return torch.device("cpu")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


def load_model(checkpoint_path: Path):
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    required_keys = {
        "model_name",
        "model_state_dict",
        "target_mean",
        "target_std",
        "node_feature_dim",
        "hidden_dim",
        "num_layers",
        "dropout",
    }

    missing = required_keys.difference(checkpoint.keys())

    if missing:
        raise KeyError(
            f"Checkpoint is missing required keys: {sorted(missing)}"
        )

    model_name = checkpoint["model_name"]
    device = choose_device(model_name)

    model = create_model(
        model_name=model_name,
        input_dim=int(checkpoint["node_feature_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        num_layers=int(checkpoint["num_layers"]),
        dropout=float(checkpoint["dropout"]),
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    return model, checkpoint, device


@torch.no_grad()
def predict_logS(
    smiles: str,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
) -> float:
    model, checkpoint, device = load_model(
        checkpoint_path
    )

    # smiles_to_graph requires a target value because it is also used
    # during training. For inference, this dummy value is never used.
    graph = smiles_to_graph(
        smiles=smiles,
        target=0.0,
        row_index=0,
    )

    batch = Batch.from_data_list(
        [graph]
    ).to(device)

    normalized_prediction = model(
        batch
    ).item()

    target_mean = float(
        checkpoint["target_mean"]
    )

    target_std = float(
        checkpoint["target_std"]
    )

    predicted_logS = (
        normalized_prediction
        * target_std
        + target_mean
    )

    return float(predicted_logS)


def main() -> None:
    args = parse_arguments()

    predicted_logS = predict_logS(
        smiles=args.smiles,
        checkpoint_path=args.checkpoint,
    )

    print("\nSolubility prediction")
    print("=" * 70)
    print("SMILES:", args.smiles)
    print(f"Predicted logS: {predicted_logS:.4f}")


if __name__ == "__main__":
    main()
