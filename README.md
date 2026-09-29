## Author

**Maryam Taherzadeh**  
Computational Scientist | AI/ML for Drug Discovery

# Molecular Solubility Prediction with Graph Neural Networks

Aqueous solubility (**logS**) prediction using Graph Convolutional Networks (**GCN**) and Graph Attention Networks (**GAT**) trained on **AqSolDB**, with random and Bemis–Murcko scaffold-based evaluation.

## Project Overview

This project predicts aqueous solubility directly from molecular structure. SMILES strings are converted into molecular graphs using RDKit and PyTorch Geometric, then used to train graph-level regression models.

The project includes:

- AqSolDB data preparation
- Molecular graph construction
- GCN and GAT training
- Evaluation using random and scaffold splits
- An exploratory Optuna hyperparameter search
- A saved scaffold-trained GCN checkpoint for downstream prediction

## Molecular Graph Representation

Each molecule is represented as a graph:

- **Nodes:** atoms
- **Edges:** covalent bonds represented in both directions
- **Target:** experimental logS

Atom features include:

- Element identity
- Atomic degree
- Formal charge
- Hybridization
- Number of attached hydrogens
- Aromaticity
- Ring membership
- Scaled atomic mass

Both models use the same molecular graph representation and preprocessing workflow.

## Model Architectures

### Graph Convolutional Network

The GCN aggregates information from neighboring atoms through graph convolutional layers. The resulting node representations are pooled into a molecular embedding and passed through a regression head to predict logS.

### Graph Attention Network

The GAT learns attention weights that allow neighboring atoms to contribute differently during message passing. The node representations are then pooled and passed through a regression head.

## Experimental Design

Four experiments were performed:

| Model | Data Split |
|---|---|
| GCN | Random |
| GAT | Random |
| GCN | Bemis–Murcko scaffold |
| GAT | Bemis–Murcko scaffold |

**Random splitting** allows structurally related compounds to appear across training, validation, and test sets.

**Scaffold splitting** groups molecules by their Bemis–Murcko scaffolds and assigns these groups to separate sets. This evaluates performance on held-out scaffolds, although molecules with different scaffolds can still share structural features.

Validation data were used for checkpoint selection and early stopping. Test data were used for final evaluation.

## Model Performance

RMSE and MAE are reported on the original logS scale.

| Model | Split | Best Epoch | Validation RMSE | Validation MAE | Validation R² | Test RMSE | Test MAE | Test R² |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| GAT | Random | 66 | 0.959 | 0.704 | 0.838 | 1.070 | 0.736 | 0.787 |
| GAT | Scaffold | 60 | 1.115 | 0.772 | 0.792 | 1.121 | 0.791 | 0.776 |
| GCN | Random | 93 | **0.903** | **0.655** | **0.856** | **0.980** | **0.662** | **0.821** |
| GCN | Scaffold | 154 | 1.057 | 0.715 | 0.813 | 1.095 | 0.751 | 0.786 |

### Results Summary

GCN achieved lower test RMSE and MAE than GAT under both splitting strategies in the reported runs.

The random-split GCN achieved the strongest numerical test performance:

- **RMSE:** 0.980
- **MAE:** 0.662
- **R²:** 0.821

The scaffold-trained GCN was retained for downstream prediction, with:

- **RMSE:** 1.095
- **MAE:** 0.751
- **R²:** 0.786

Scaffold evaluation provides evidence about performance on held-out scaffolds. It does not, by itself, establish that the scaffold-trained model is more accurate for every new molecule.

## Training Strategy

The main experiments used the following training setup:

| Setting | Value |
|---|---|
| Loss function | Mean Squared Error |
| Optimizer | AdamW |
| Initial learning rate | 0.001 |
| Weight decay | 0.00001 |
| Hidden dimension | 128 |
| Graph layers | 3 |
| Dropout | 0.20 |
| Batch size | 64 |
| Maximum epochs | 300 |
| Early-stopping patience | 40 |
| Random seed | 42 |

Experimental logS values were standardized using statistics calculated from the training set only.

Validation RMSE was used for model monitoring and early stopping. The best model checkpoint was restored before final evaluation, and performance metrics were calculated on the original logS scale.

Experiment-specific settings are recorded in the `configuration.json` files within `results/`.

## Exploratory Hyperparameter Search

An exploratory Optuna search was performed for the scaffold-split GCN.

| Configuration | Validation RMSE |
|---|---:|
| Baseline GCN | **1.056744** |
| Best Optuna trial | 1.057174 |

The search did not improve validation RMSE, so the Optuna configuration was not used in the final model. The baseline scaffold-trained GCN was retained.

The optimization notebook is included as a record of this experiment.

## Evaluation Metrics

- **RMSE:** measures prediction error while giving greater weight to larger errors.
- **MAE:** measures the average absolute difference between experimental and predicted logS.
- **R²:** measures performance relative to predicting the evaluation-set mean; higher values indicate better agreement.

## Workflow

1. Clean and prepare AqSolDB data.
2. Create random and scaffold-based data splits.
3. Convert SMILES into molecular graphs.
4. Calculate target-standardization statistics from the training set.
5. Train GCN and GAT models.
6. Select checkpoints using validation RMSE.
7. Evaluate predictions on the held-out test sets.
8. Compare results and retain the scaffold-trained GCN checkpoint.

## Repository Contents

| Path | Description |
|---|---|
| `data/processed/` | Cleaned AqSolDB data, split datasets, and split summaries |
| `src/prepare_aqsoldb.py` | Data preparation |
| `src/train_gnn.py` | GNN training and evaluation |
| `src/optimize_gcn.py` | Exploratory GCN hyperparameter search |
| `notebooks/01_GNN_Solubility_Training.ipynb` | Training workflow and model evaluation |
| `notebooks/02_GCN_Optimization.ipynb` | Optuna experiments |
| `models/gcn_scaffold_best.pt` | Saved baseline scaffold-trained GCN checkpoint |
| `results/gcn_random/` | Random-split GCN results |
| `results/gcn_scaffold/` | Scaffold-split GCN results |
| `results/gat_random/` | Random-split GAT results |
| `results/gat_scaffold/` | Scaffold-split GAT results |
| `requirements.txt` | Python dependencies |

Each main experiment directory contains:

- `configuration.json`
- `metrics.json`
- `training_history.csv`
- `train_predictions.csv`
- `validation_predictions.csv`
- `test_predictions.csv`

## Getting Started

Clone the repository and enter the project directory:

```bash
git clone https://github.com/Maryam-Taherzadeh/Molecular-Solubility-GNN.git
cd Molecular-Solubility-GNN
```

Install the dependencies in your Python environment:

```bash
pip install -r requirements.txt
```

Open `notebooks/01_GNN_Solubility_Training.ipynb` in a Jupyter-compatible environment to review the training and evaluation workflow.

The separate optimization experiments are documented in `notebooks/02_GCN_Optimization.ipynb`.

## Limitations

- The reported comparison uses a single random seed and does not measure variability across repeated runs.
- Random and scaffold experiments use different test sets, so their metrics reflect differences in both training and evaluation data.
- Scaffold splitting does not eliminate all structural similarity between sets.
- Aggregate test metrics do not provide uncertainty estimates for individual predictions.
- The small Optuna search does not establish that further tuning would be ineffective.

## Planned Extension

A Streamlit application is planned to allow users to:

- Enter a SMILES string
- Validate and visualize the molecular structure
- Convert the molecule into a graph
- Generate a logS prediction using the saved GCN model

## Technology Stack

- Python
- PyTorch
- PyTorch Geometric
- RDKit
- pandas
- NumPy
- scikit-learn
- Optuna

