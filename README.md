# Molecular-Solubility-GNN
Aqueous solubility prediction using GCN and GAT models with random and scaffold-based evaluation on AqSolDB.
# 🧪 Molecular Solubility Prediction with Graph Neural Networks

A molecular machine learning project for predicting aqueous solubility (**logS**) directly from molecular structure using **Graph Neural Networks (GNNs)** trained on the **AqSolDB** dataset.

This project compares **Graph Convolutional Networks (GCN)** and **Graph Attention Networks (GAT)** under both **random** and **Bemis–Murcko scaffold-based** data splitting strategies to evaluate predictive performance and molecular generalization.

---

## 🔬 Project Overview

Aqueous solubility is an important molecular property in drug discovery because it influences compound behavior, formulation, exposure, and downstream candidate prioritization.

In this project, molecular structures represented as SMILES are converted into molecular graphs using **RDKit** and **PyTorch Geometric**.

Each molecule is represented as:

- **Atoms → graph nodes**
- **Covalent bonds → bidirectional graph edges**
- **Experimental logS → graph-level regression target**

Two graph neural network architectures were developed and compared:

- Graph Convolutional Network (**GCN**)
- Graph Attention Network (**GAT**)

---

## 🧬 Molecular Graph Representation

Each molecule is converted from SMILES into a PyTorch Geometric graph.

### Atom Features

Each atom is represented using:

- Element identity
- Atomic degree
- Formal charge
- Hybridization
- Number of attached hydrogens
- Aromaticity
- Ring membership
- Scaled atomic mass

Covalent bonds are represented as bidirectional graph edges.

---

## 🧠 GNN Architectures

### Graph Convolutional Network (GCN)

The GCN architecture performs neighborhood aggregation using graph convolutional layers to learn molecular representations from local atomic environments.

The learned graph representation is pooled into a molecular-level embedding and passed through a regression head to predict aqueous solubility.

### Graph Attention Network (GAT)

The GAT architecture uses learned attention coefficients during message passing, allowing neighboring atoms to contribute differently to the learned molecular representation.

Both GCN and GAT models use the same molecular graph representation and preprocessing workflow to enable a consistent comparison.

---

## 🧪 Experimental Design

Four experiments were performed:

1. **GCN — Random Split**
2. **GAT — Random Split**
3. **GCN — Bemis–Murcko Scaffold Split**
4. **GAT — Bemis–Murcko Scaffold Split**

Random splitting evaluates performance when structurally related compounds may occur across training and test sets.

The scaffold split provides a more stringent evaluation by separating molecules according to their **Bemis–Murcko molecular scaffolds**, providing a stronger test of generalization to structurally distinct compounds.

---

## 📊 Model Performance

| Model | Split | Best Epoch | Validation RMSE | Validation MAE | Validation R² | Test RMSE | Test MAE | Test R² |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| GAT | Random | 66 | 0.959 | 0.704 | 0.838 | 1.070 | 0.736 | 0.787 |
| GAT | Scaffold | 60 | 1.115 | 0.772 | 0.792 | 1.121 | 0.791 | 0.776 |
| GCN | Random | 93 | **0.903** | **0.655** | **0.856** | **0.980** | **0.662** | **0.821** |
| GCN | Scaffold | 154 | 1.057 | 0.715 | 0.813 | 1.095 | 0.751 | 0.786 |

### Best Numerical Performance

The **GCN with random splitting** achieved the strongest numerical test performance:

- **Test R²:** 0.821
- **Test RMSE:** 0.980
- **Test MAE:** 0.662

### Selected Model for Downstream Prediction

The **GCN with Bemis–Murcko scaffold splitting** was retained for downstream solubility prediction because scaffold-based evaluation provides a more stringent assessment of generalization to structurally distinct molecules.

Its independent test performance was:

- **Test R²:** 0.786
- **Test RMSE:** 1.095
- **Test MAE:** 0.751

---

## ⚙️ Training Strategy

The GNN models were trained for graph-level regression using:

- **Loss:** Mean Squared Error (MSE)
- **Optimizer:** AdamW
- **Initial learning rate:** 1 × 10⁻³
- **Weight decay:** 1 × 10⁻⁵
- **Hidden dimension:** 128
- **Graph layers:** 3
- **Dropout:** 0.20
- **Batch size:** 64
- **Maximum epochs:** 300
- **Early-stopping patience:** 40
- **Random seed:** 42

Validation RMSE was used for model monitoring and early stopping.

The best model checkpoint was restored before final evaluation.

Experimental logS values were standardized using statistics calculated from the **training set only**, and final performance metrics were reported on the original logS scale.

---

## 📏 Evaluation Metrics

Model performance was evaluated using:

### Root Mean Squared Error (RMSE)

Measures the typical magnitude of prediction error while giving larger errors greater weight.

### Mean Absolute Error (MAE)

Measures the average absolute difference between experimental and predicted logS.

### Coefficient of Determination (R²)

Measures the fraction of variance in experimental solubility explained by the model.

---

## 🔄 Workflow

```text
AqSolDB
    │
    ▼
Data Preprocessing
    │
    ▼
SMILES
    │
    ▼
RDKit Molecular Graph Construction
    │
    ▼
Atom Features + Bond Connectivity
    │
    ├───────────────┐
    ▼               ▼
   GCN             GAT
    │               │
    └───────┬───────┘
            ▼
   Random / Scaffold Split
            │
            ▼
      Model Training
            │
            ▼
 Validation & Early Stopping
            │
            ▼
       Test Evaluation
            │
            ▼
     Model Comparison
            │
            ▼
  Scaffold GCN Selection
            │
            ▼
     logS Prediction
```

---

## 🚀 Interactive Solubility Predictor

An interactive **Streamlit application** will use the selected scaffold-based GCN model for molecular solubility prediction.

The application will allow users to:

- Enter a molecular SMILES string
- Validate the molecular structure
- Visualize the molecule
- Convert the molecule into a graph
- Run GCN inference
- Obtain predicted aqueous solubility (**logS**)

---

## 📁 Repository Structure

```text
molecular-solubility-gnn/
│
├── README.md
├── app.py
├── requirements.txt
│
├── src/
│   ├── model.py
│   ├── featurization.py
│   ├── predict.py
│   └── train_gnn.py
│
├── models/
│   └── gcn_scaffold_best.pt
│
├── notebooks/
│   └── GNN_Solubility_Modeling.ipynb
│
├── figures/
│   ├── model_comparison.png
│   └── predicted_vs_experimental.png
│
└── data/
    └── README.md
```

---

## 🛠️ Technology Stack

- **Python**
- **PyTorch**
- **PyTorch Geometric**
- **RDKit**
- **Pandas**
- **NumPy**
- **scikit-learn**
- **Streamlit**

---

## 💡 Key Takeaway

The experiments demonstrate the importance of evaluating molecular machine-learning models beyond random train/test splitting.

Although the random-split GCN achieved the highest numerical performance (**R² = 0.821**), the scaffold-based evaluation provides a more challenging assessment of generalization to structurally distinct molecules.

For this reason, the **scaffold-based GCN** was selected as the downstream aqueous-solubility prediction model.

---

## 👩‍💻 Author

**Maryam Taherzadeh**  
Computational Scientist | AI/ML for Drug Discovery
