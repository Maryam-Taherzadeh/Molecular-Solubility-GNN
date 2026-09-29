# ============================================================
# OPTIMIZE GCN HYPERPARAMETERS WITH OPTUNA
#
# This script:
# 1. Loads the scaffold training and validation datasets
# 2. Converts molecules into graph objects
# 3. Uses Optuna to select GCN hyperparameters
# 4. Trains one GCN model for each Optuna trial
# 5. Uses validation RMSE as the optimization objective
# 6. Uses early stopping inside every trial
# 7. Saves all trial results and the best parameters
#
# Important:
# - The scaffold test set is not loaded or evaluated here.
# - Hyperparameters must be selected using validation data.
# - The test set should be evaluated only after optimization.
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

# Handle command-line arguments
import argparse

# Save results in JSON format
import json

# Release unused memory between trials
import gc

# Handle file and folder paths
from pathlib import Path

# Perform numerical operations
import numpy as np

# Load tabular datasets
import pandas as pd

# Train neural networks
import torch

# Create graph data loaders
from torch_geometric.loader import DataLoader

# Optimize hyperparameters
import optuna

# Import the existing GNN training functions
from train_gnn import (
    NODE_FEATURE_DIM,
    create_model,
    dataframe_to_graphs,
    evaluate,
    load_split_dataframe,
    normalize_graph_targets,
    set_random_seed,
    train_one_epoch,
)


# ============================================================
# 2. COMMAND-LINE ARGUMENTS
# ============================================================

def parse_arguments():
    """
    Read Optuna settings from the command line.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Optimize scaffold-split GCN hyperparameters "
            "using Optuna."
        )
    )

    # Number of Optuna configurations to test
    parser.add_argument(
        "--trials",
        type=int,
        default=20,
        help="Number of Optuna trials.",
    )

    # Maximum training epochs for each trial
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Maximum epochs for each trial.",
    )

    # Stop a trial when validation RMSE does not improve
    parser.add_argument(
        "--patience",
        type=int,
        default=25,
        help="Early-stopping patience for each trial.",
    )

    # Fixed random seed
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    # Keep zero initially on macOS
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker count.",
    )

    return parser.parse_args()


# ============================================================
# 3. PROJECT PATHS
# ============================================================

# Find the main project directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Define the processed-data directory
PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)


# Define the scaffold training dataset
TRAIN_PATH = (
    PROCESSED_DIR
    / "aqsoldb_scaffold_train.csv"
)


# Define the scaffold validation dataset
VALIDATION_PATH = (
    PROCESSED_DIR
    / "aqsoldb_scaffold_validation.csv"
)


# Define the output directory for Optuna
OPTUNA_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "gnn_optuna"
    / "gcn_scaffold"
)


# Create the Optuna output directory
OPTUNA_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# Define the persistent Optuna database
STUDY_DATABASE_PATH = (
    OPTUNA_OUTPUT_DIR
    / "optuna_study.db"
)


# Define the best-parameter output file
BEST_PARAMETERS_PATH = (
    OPTUNA_OUTPUT_DIR
    / "best_parameters.json"
)


# Define the complete trial-results output file
TRIAL_RESULTS_PATH = (
    OPTUNA_OUTPUT_DIR
    / "trial_results.csv"
)


# ============================================================
# 4. SELECT THE COMPUTING DEVICE
# ============================================================

def select_device():
    """
    Select CUDA, Apple MPS, or CPU.

    GCN operations are normally supported on Apple MPS.
    """

    # Use an NVIDIA GPU when CUDA is available
    if torch.cuda.is_available():
        return torch.device(
            "cuda"
        )

    # Use the Apple GPU when MPS is available
    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device(
            "mps"
        )

    # Use CPU when no GPU backend is available
    return torch.device(
        "cpu"
    )


# ============================================================
# 5. RELEASE MEMORY BETWEEN TRIALS
# ============================================================

def clear_device_memory():
    """
    Release Python and device memory after each Optuna trial.
    """

    # Run Python garbage collection
    gc.collect()

    # Release unused CUDA memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Release unused Apple MPS memory when supported
    if (
        hasattr(torch, "mps")
        and hasattr(torch.mps, "empty_cache")
    ):
        torch.mps.empty_cache()


# ============================================================
# 6. LOAD AND PREPARE THE DATA ONCE
# ============================================================

def prepare_datasets(seed):
    """
    Load scaffold train and validation datasets.

    Convert them to graphs and normalize logS using only
    training-set statistics.
    """

    # Use a reproducible random seed
    set_random_seed(
        seed
    )

    # Load the scaffold training dataset
    train_df = load_split_dataframe(
        TRAIN_PATH,
        "scaffold training dataset",
    )

    # Load the scaffold validation dataset
    validation_df = load_split_dataframe(
        VALIDATION_PATH,
        "scaffold validation dataset",
    )

    # Calculate the training-target mean
    target_mean = float(
        train_df["logS"].mean()
    )

    # Calculate the population standard deviation
    target_std = float(
        train_df["logS"].std(
            ddof=0
        )
    )

    # Validate normalization values
    if (
        not np.isfinite(target_mean)
        or not np.isfinite(target_std)
        or target_std <= 0
    ):
        raise ValueError(
            "Invalid training-target normalization statistics."
        )

    # Convert training molecules into graphs
    train_graphs = dataframe_to_graphs(
        train_df,
        "Optuna scaffold training",
    )

    # Convert validation molecules into graphs
    validation_graphs = dataframe_to_graphs(
        validation_df,
        "Optuna scaffold validation",
    )

    # Normalize training targets
    normalize_graph_targets(
        train_graphs,
        target_mean,
        target_std,
    )

    # Normalize validation targets using training statistics
    normalize_graph_targets(
        validation_graphs,
        target_mean,
        target_std,
    )

    return (
        train_graphs,
        validation_graphs,
        target_mean,
        target_std,
    )


# ============================================================
# 7. CREATE THE OPTUNA OBJECTIVE
# ============================================================

def create_objective(
    train_graphs,
    validation_graphs,
    target_mean,
    target_std,
    device,
    maximum_epochs,
    early_stopping_patience,
    seed,
    num_workers,
):
    """
    Create the function that Optuna evaluates for each trial.
    """

    def objective(trial):
        """
        Train one GCN configuration and return its best
        scaffold validation RMSE.
        """

        # Use the same seed so trials are fairly comparable
        set_random_seed(
            seed
        )

        # Select the hidden representation size
        hidden_dim = trial.suggest_categorical(
            "hidden_dim",
            [
                64,
                128,
                256,
            ],
        )

        # Select the number of graph-convolution layers
        num_layers = trial.suggest_int(
            "num_layers",
            2,
            5,
        )

        # Select the dropout probability
        dropout = trial.suggest_float(
            "dropout",
            0.10,
            0.50,
            step=0.05,
        )

        # Select the AdamW learning rate
        learning_rate = trial.suggest_float(
            "learning_rate",
            1.0e-4,
            3.0e-3,
            log=True,
        )

        # Select the AdamW weight-decay coefficient
        weight_decay = trial.suggest_float(
            "weight_decay",
            1.0e-7,
            1.0e-3,
            log=True,
        )

        # Select the molecular-graph batch size
        batch_size = trial.suggest_categorical(
            "batch_size",
            [
                32,
                64,
                128,
            ],
        )

        # Create a reproducible DataLoader generator
        loader_generator = torch.Generator()

        loader_generator.manual_seed(
            seed
        )

        # Create the shuffled training DataLoader
        train_loader = DataLoader(
            train_graphs,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            generator=loader_generator,
        )

        # Create the validation DataLoader
        validation_loader = DataLoader(
            validation_graphs,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        # Create the GCN model
        model = create_model(
            model_name="gcn",
            input_dim=NODE_FEATURE_DIM,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        # Move the model to the selected device
        model = model.to(
            device
        )

        # Use AdamW to optimize neural-network weights
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Reduce the learning rate when validation RMSE plateaus
        scheduler = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=8,
                min_lr=1.0e-6,
            )
        )

        # Store the best validation RMSE for this trial
        best_validation_rmse = float(
            "inf"
        )

        # Store the best epoch for this trial
        best_epoch = 0

        # Count epochs without validation improvement
        epochs_without_improvement = 0

        try:
            # Train this configuration
            for epoch in range(
                1,
                maximum_epochs + 1,
            ):
                # Train the GCN for one epoch
                train_one_epoch(
                    model=model,
                    loader=train_loader,
                    optimizer=optimizer,
                    device=device,
                )

                # Evaluate on scaffold validation data
                (
                    validation_metrics,
                    _,
                    _,
                    _,
                ) = evaluate(
                    model=model,
                    loader=validation_loader,
                    device=device,
                    target_mean=target_mean,
                    target_std=target_std,
                )

                # Read validation RMSE in the original logS scale
                validation_rmse = float(
                    validation_metrics["rmse"]
                )

                # Update the learning-rate scheduler
                scheduler.step(
                    validation_rmse
                )

                # Report this epoch to Optuna
                trial.report(
                    validation_rmse,
                    step=epoch,
                )

                # Check whether validation RMSE improved
                if (
                    validation_rmse
                    < best_validation_rmse
                    - 1.0e-6
                ):
                    best_validation_rmse = (
                        validation_rmse
                    )

                    best_epoch = epoch

                    epochs_without_improvement = 0

                else:
                    epochs_without_improvement += 1

                # Allow Optuna to stop weak trials early
                if trial.should_prune():
                    raise optuna.TrialPruned()

                # Stop this trial after a validation plateau
                if (
                    epochs_without_improvement
                    >= early_stopping_patience
                ):
                    break

            # Save useful trial metadata
            trial.set_user_attr(
                "best_epoch",
                best_epoch,
            )

            trial.set_user_attr(
                "device",
                str(device),
            )

            # Return the score Optuna should minimize
            return best_validation_rmse

        finally:
            # Remove trial objects from memory
            del model
            del optimizer
            del scheduler
            del train_loader
            del validation_loader

            # Release unused device memory
            clear_device_memory()

    return objective


# ============================================================
# 8. RUN THE OPTUNA STUDY
# ============================================================

def main():
    """
    Run GCN hyperparameter optimization.
    """

    # Read command-line settings
    arguments = parse_arguments()

    # Select the training device
    device = select_device()

    print("\nOptuna configuration")
    print("=" * 70)

    print(
        "Model: GCN"
    )

    print(
        "Split: scaffold"
    )

    print(
        "Device:",
        device,
    )

    print(
        "Requested trials:",
        arguments.trials,
    )

    print(
        "Maximum epochs per trial:",
        arguments.epochs,
    )

    print(
        "Early-stopping patience:",
        arguments.patience,
    )

    # Load and prepare graph datasets once
    (
        train_graphs,
        validation_graphs,
        target_mean,
        target_std,
    ) = prepare_datasets(
        arguments.seed
    )

    print("\nPrepared datasets")
    print("=" * 70)

    print(
        "Training graphs:",
        len(train_graphs),
    )

    print(
        "Validation graphs:",
        len(validation_graphs),
    )

    print(
        "Training logS mean:",
        target_mean,
    )

    print(
        "Training logS standard deviation:",
        target_std,
    )

    # Create a reproducible Optuna sampler
    sampler = optuna.samplers.TPESampler(
        seed=arguments.seed
    )

    # Stop unpromising trials after several epochs
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=20,
        interval_steps=5,
    )

    # Store the study in a local SQLite database
    storage_url = (
        f"sqlite:///{STUDY_DATABASE_PATH}"
    )

    # Create or resume the Optuna study
    study = optuna.create_study(
        study_name="gcn_scaffold",
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=storage_url,
        load_if_exists=True,
    )

    # Create the objective function
    objective = create_objective(
        train_graphs=train_graphs,
        validation_graphs=validation_graphs,
        target_mean=target_mean,
        target_std=target_std,
        device=device,
        maximum_epochs=arguments.epochs,
        early_stopping_patience=arguments.patience,
        seed=arguments.seed,
        num_workers=arguments.num_workers,
    )

    # Run Optuna trials
    study.optimize(
        objective,
        n_trials=arguments.trials,
        gc_after_trial=True,
    )

    # Convert all trial results into a dataframe
    trial_results_df = study.trials_dataframe(
        attrs=(
            "number",
            "value",
            "state",
            "params",
            "user_attrs",
        )
    )

    # Save all trial results
    trial_results_df.to_csv(
        TRIAL_RESULTS_PATH,
        index=False,
    )

    # Combine the best parameters with the best validation score
    best_result = {
        "study_name": study.study_name,
        "best_trial_number": study.best_trial.number,
        "best_validation_rmse": study.best_value,
        "best_parameters": study.best_params,
        "best_epoch": study.best_trial.user_attrs.get(
            "best_epoch"
        ),
        "device": str(device),
        "number_of_completed_trials": len(
            study.trials
        ),
    }

    # Save the best Optuna result
    with BEST_PARAMETERS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            best_result,
            file,
            indent=2,
        )

    # Print the best result
    print("\nBest Optuna result")
    print("=" * 70)

    print(
        "Best trial:",
        study.best_trial.number,
    )

    print(
        "Best validation RMSE:",
        f"{study.best_value:.4f}",
    )

    print(
        "Best epoch:",
        study.best_trial.user_attrs.get(
            "best_epoch"
        ),
    )

    print("\nBest parameters")

    for parameter_name, parameter_value in (
        study.best_params.items()
    ):
        print(
            f"{parameter_name}: {parameter_value}"
        )

    # Print the command for final model training
    best_parameters = study.best_params

    print("\nFinal training command")
    print("=" * 70)

    print(
        "python -u src/models/train_gnn.py "
        "--model gcn "
        "--split scaffold "
        f"--hidden-dim {best_parameters['hidden_dim']} "
        f"--num-layers {best_parameters['num_layers']} "
        f"--dropout {best_parameters['dropout']} "
        f"--learning-rate {best_parameters['learning_rate']} "
        f"--weight-decay {best_parameters['weight_decay']} "
        f"--batch-size {best_parameters['batch_size']} "
        "--epochs 300 "
        "--patience 40"
    )

    print("\nSaved Optuna files")
    print("=" * 70)

    print(
        STUDY_DATABASE_PATH
    )

    print(
        TRIAL_RESULTS_PATH
    )

    print(
        BEST_PARAMETERS_PATH
    )

    print(
        "\nOptuna optimization completed successfully."
    )


# ============================================================
# 9. RUN THE SCRIPT
# ============================================================

if __name__ == "__main__":
    main()