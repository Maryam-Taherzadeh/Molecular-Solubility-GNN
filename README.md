# Molecular Solubility Prediction with Graph Neural Networks

Aqueous solubility (**logS**) prediction using Graph Convolutional Networks (**GCN**) and Graph Attention Networks (**GAT**) trained on **AqSolDB**, with both random and Bemis–Murcko scaffold-based evaluation.

The repository includes the complete model-development workflow as well as a deployed **Streamlit application** for single-molecule inference using the scaffold-trained GCN model.

---

## Live Demo

An interactive Streamlit application is available for molecular solubility prediction.

**Streamlit App:**  
`ADD_YOUR_STREAMLIT_APP_URL_HERE`

The application supports:

- Molecule name or SMILES input
- PubChem name-to-SMILES lookup
- GCN-based logS prediction
- Molecular descriptor calculation
- Comparison with predictions from the held-out scaffold test set
- Model performance display
- Downloadable PDF prediction reports

---

## Project Overview

Aqueous solubility is an important molecular property in medicinal chemistry and drug discovery because poor solubility can affect compound formulation, bioavailability, and experimental usability.

This project predicts aqueous solubility directly from molecular structure.

SMILES strings are converted into molecular graphs using **RDKit** and **PyTorch Geometric**, and graph neural networks are trained to perform graph-level regression on experimental logS values from **AqSolDB**.

The project includes:

- AqSolDB data preparation
- Molecular graph construction from SMILES
- GCN and GAT model development
- Random and scaffold-based dataset splitting
- Training-target standardization
- Early stopping and checkpoint selection
- Model evaluation using RMSE, MAE, and R²
- Exploratory Optuna hyperparameter optimization
- Saved scaffold-trained GCN checkpoint
- Single-molecule inference pipeline
- Streamlit deployment for interactive predictions

---

## Molecular Graph Representation

Each molecule is represented as a graph:

- **Nodes:** atoms
- **Edges:** covalent bonds represented in both directions
- **Target:** experimental aqueous solubility, logS

Each atom is represented using features describing its local chemical environment.

### Atom Features

The current graph representation includes:

- Element identity
- Atomic degree
- Formal charge
- Hybridization
- Number of attached hydrogens
- Aromaticity
- Ring membership
- Scaled atomic mass

The resulting node feature dimension is:

**40 features per atom**

GCN and GAT models use the same molecular graph representation and preprocessing workflow.

---

## Model Architectures

### Graph Convolutional Network

The GCN applies graph convolutional layers to aggregate information from neighboring atoms.

The general architecture is:

```text
Molecular Graph
      ↓
GCN Layers
      ↓
Batch Normalization
      ↓
ReLU
      ↓
Dropout
      ↓
Global Mean Pooling
      ↓
Molecular Embedding
      ↓
Regression Head
      ↓
Predicted logS
```

Node representations are pooled into a fixed-length molecular embedding using global mean pooling before regression.

### Graph Attention Network

The GAT uses attention-based message passing so that neighboring atoms can contribute differently to the learned node representation.

After the graph-attention layers, node embeddings are pooled into a molecular representation and passed through a regression head to predict logS.

---

## Experimental Design

Four primary experiments were performed:

| Model | Data Split |
|---|---|
| GCN | Random |
| GAT | Random |
| GCN | Bemis–Murcko scaffold |
| GAT | Bemis–Murcko scaffold |

### Random Split

Random splitting assigns molecules independently across training, validation, and test sets.

This can result in structurally related compounds appearing in different subsets.

### Scaffold Split

Scaffold splitting groups compounds using their **Bemis–Murcko scaffolds** before assigning groups to training, validation, and test sets.

This provides a more challenging evaluation because the test set contains scaffolds that were held out during training.

Scaffold splitting does not guarantee complete structural dissimilarity between all molecules, but it provides a stronger test of structural generalization than a conventional random split.

### Dataset Usage

Validation data were used for:

- model monitoring
- early stopping
- checkpoint selection

The test sets were reserved for final evaluation.

---

## Model Performance

RMSE and MAE are reported on the original logS scale.

| Model | Split | Best Epoch | Validation RMSE | Validation MAE | Validation R² | Test RMSE | Test MAE | Test R² |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| GAT | Random | 66 | 0.959 | 0.704 | 0.838 | 1.070 | 0.736 | 0.787 |
| GAT | Scaffold | 60 | 1.115 | 0.772 | 0.792 | 1.121 | 0.791 | 0.776 |
| GCN | Random | 93 | **0.903** | **0.655** | **0.856** | **0.980** | **0.662** | **0.821** |
| GCN | Scaffold | 154 | 1.057 | 0.715 | 0.813 | 1.095 | 0.751 | 0.786 |

---

## Results Summary

GCN achieved lower test RMSE and MAE than GAT under both splitting strategies in the reported experiments.

### Random-Split GCN

The random-split GCN achieved the strongest numerical test performance:

- **Test RMSE:** 0.980
- **Test MAE:** 0.662
- **Test R²:** 0.821

### Scaffold-Split GCN

The scaffold-trained GCN achieved:

- **Validation RMSE:** 1.0567
- **Test RMSE:** 1.0946
- **Test MAE:** 0.7506
- **Test R²:** 0.7861

This model was retained for downstream inference and is the model currently used by the Streamlit application.

Scaffold evaluation provides evidence about performance on held-out scaffolds. It does not imply that the model will achieve the same level of accuracy for every new molecule.

---

## Deployed GCN Configuration

The scaffold-trained GCN used by the deployed application has the following configuration:

| Setting | Value |
|---|---:|
| Model | GCN |
| Data split | Scaffold |
| Hidden dimension | 256 |
| Graph layers | 3 |
| Dropout | 0.30 |
| Learning rate | 0.0005 |
| Weight decay | 0.00001 |
| Batch size | 64 |
| Maximum epochs | 300 |
| Early-stopping patience | 40 |
| Random seed | 42 |
| Node feature dimension | 40 |
| Best epoch | 154 |

The scaffold experiment contained:

| Dataset | Molecules |
|---|---:|
| Training | 7,479 |
| Validation | 934 |
| Test | 936 |

Experimental logS values were standardized using statistics calculated only from the training set.

For the deployed scaffold-trained GCN:

```text
Training target mean = -3.042688
Training target std  =  2.284667
```

Model predictions are inverse-transformed back to the original logS scale before reporting results.

Experiment-specific configurations are stored in the corresponding `configuration.json` files under `results/`.

---

## Training Strategy

The training pipeline performs the following steps:

1. Load the prepared AqSolDB split.
2. Convert SMILES strings into PyTorch Geometric molecular graphs.
3. Calculate target normalization statistics from the training set only.
4. Standardize the training targets.
5. Train the selected graph neural network.
6. Monitor validation RMSE.
7. Apply early stopping when validation performance stops improving.
8. Save the best-performing checkpoint.
9. Restore the best checkpoint.
10. Evaluate train, validation, and test predictions on the original logS scale.

The primary loss function is:

**Mean Squared Error**

Optimization is performed using:

**AdamW**

---

## Exploratory Hyperparameter Search

An exploratory **Optuna** hyperparameter search was also performed for the scaffold-split GCN.

| Configuration | Validation RMSE |
|---|---:|
| Baseline GCN | **1.056744** |
| Best Optuna trial | 1.057174 |

The exploratory search did not improve validation RMSE relative to the baseline scaffold-trained model.

The baseline model was therefore retained as the final scaffold GCN.

The optimization workflow is preserved in the repository for reproducibility and experimentation.

---

## Evaluation Metrics

Three regression metrics are reported.

### RMSE

**Root Mean Squared Error**

RMSE measures prediction error while assigning greater weight to larger errors.

Lower values are better.

### MAE

**Mean Absolute Error**

MAE measures the average absolute difference between experimental and predicted logS values.

Lower values are better.

### R²

**Coefficient of Determination**

R² measures predictive performance relative to predicting the mean value of the evaluation set.

Higher values indicate stronger agreement between predicted and experimental values.

---

## Inference Workflow

The saved scaffold-trained GCN can be used without retraining.

The inference pipeline is:

```text
Molecule name or SMILES
          ↓
SMILES resolution / validation
          ↓
RDKit molecule
          ↓
Atom feature extraction
          ↓
PyTorch Geometric graph
          ↓
Saved GCN checkpoint
          ↓
Normalized prediction
          ↓
Inverse target transformation
          ↓
Predicted logS
```

For molecule names, the Streamlit application can query **PubChem** to obtain the corresponding SMILES representation.

---

## Interactive Streamlit Application

The project includes an interactive Streamlit interface in:

```text
app.py
```

The application loads the saved scaffold-trained GCN model:

```text
outputs/gnn_experiments/gcn_scaffold/best_model.pt
```

### Application Features

Users can:

- Enter a SMILES string
- Enter a molecule name
- Resolve molecule names through PubChem
- Validate molecular structures using RDKit
- Convert the molecule into graph features matching the training pipeline
- Generate a GCN-based logS prediction
- View model-derived approximate molar solubility
- View qualitative solubility classification
- Calculate molecular descriptors
- Compare the prediction against real scaffold-test model predictions
- Review the model's held-out test performance
- Download a PDF prediction report

### Molecular Descriptors

The application calculates:

- Molecular weight
- LogP
- TPSA
- Hydrogen-bond donors
- Hydrogen-bond acceptors
- Rotatable bonds

### Approximate Molar Solubility

The application also calculates:

```text
Molar solubility ≈ 10^(predicted logS)
```

This value is shown as a **model-derived approximation** and should be interpreted together with the model's test error.

---

## Repository Structure

| Path | Description |
|---|---|
| `app.py` | Streamlit inference application |
| `data/processed/` | Cleaned AqSolDB datasets and dataset splits |
| `src/prepare_aqsoldb.py` | AqSolDB preparation workflow |
| `src/train_gnn.py` | GCN/GAT training and evaluation |
| `src/evaluate_gnn.py` | Evaluation of the saved scaffold GCN checkpoint |
| `src/predict_gnn.py` | Single-molecule inference script |
| `src/optimize_gcn.py` | Exploratory Optuna hyperparameter optimization |
| `notebooks/01_GNN_Solubility_Training.ipynb` | Training and evaluation workflow |
| `notebooks/02_GCN_Optimization.ipynb` | Hyperparameter optimization experiments |
| `outputs/gnn_experiments/gcn_scaffold/best_model.pt` | GCN checkpoint used by the deployed app |
| `results/gcn_random/` | Random-split GCN results |
| `results/gcn_scaffold/` | Scaffold-split GCN results |
| `results/gat_random/` | Random-split GAT results |
| `results/gat_scaffold/` | Scaffold-split GAT results |
| `requirements.txt` | Python dependencies |

Each primary experiment directory contains files such as:

```text
configuration.json
metrics.json
training_history.csv
train_predictions.csv
validation_predictions.csv
test_predictions.csv
```

---

## Running the Project Locally

Clone the repository:

```bash
git clone https://github.com/Maryam-Taherzadeh/Molecular-Solubility-GNN.git
```

Enter the project directory:

```bash
cd Molecular-Solubility-GNN
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the Streamlit Application

Start the application with:

```bash
streamlit run app.py
```

Then open the local Streamlit URL displayed in the terminal.

---

## Single-Molecule Prediction from the Command Line

The saved model can also be used through the prediction script.

Example:

```bash
python src/predict_gnn.py --smiles "CCO"
```

Example input:

```text
CCO
```

corresponds to ethanol.

The script:

1. loads the saved GCN checkpoint
2. generates molecular graph features
3. performs inference
4. reverses target standardization
5. reports the predicted logS value

---

## Evaluate the Saved GCN Model

The stored checkpoint can be evaluated independently of model training using:

```bash
python src/evaluate_gnn.py
```

This allows the saved model to be evaluated without retraining the GNN.

---

## Technology Stack

### Machine Learning

- Python
- PyTorch
- PyTorch Geometric
- scikit-learn
- Optuna

### Cheminformatics

- RDKit
- AqSolDB

### Data Analysis

- pandas
- NumPy
- Matplotlib

### Application and Deployment

- Streamlit
- PubChem PUG REST API
- Requests
- FPDF2
- Git
- GitHub

---

## Limitations

Several limitations should be considered when interpreting the reported results.

- The primary model comparison uses a single random seed and therefore does not quantify variability across repeated runs.
- Random-split and scaffold-split experiments use different test sets, so their metrics reflect both training and evaluation-set differences.
- Scaffold splitting reduces direct scaffold overlap but does not eliminate all structural similarity between datasets.
- Aggregate test metrics do not provide molecule-specific prediction uncertainty.
- The exploratory Optuna search was limited and does not establish that additional optimization could not improve performance.
- Predictions for molecules far outside the training distribution may be less reliable.
- The conversion from predicted logS to molar solubility is derived directly from the model prediction and inherits the model's prediction uncertainty.
- The model should be considered a computational prioritization tool rather than a substitute for experimental solubility measurements.

---

## Future Work

Potential extensions include:

- Applicability-domain estimation
- Prediction uncertainty estimation
- Repeated scaffold-split evaluation
- Ensemble GNN models
- Additional molecular descriptors
- Edge/bond features in message passing
- Molecular fingerprints combined with GNN embeddings
- Transformer-based molecular representations
- Additional solubility datasets
- External validation
- Explainability and atom-level attribution
- Hyperparameter optimization at larger scale

---

## Author

**Maryam Taherzadeh**

Computational Scientist | AI/ML for Drug Discovery

GitHub:  
https://github.com/Maryam-Taherzadeh
