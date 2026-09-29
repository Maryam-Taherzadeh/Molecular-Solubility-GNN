# ============================================================
# EVALUATE A SAVED GNN CHECKPOINT
# ============================================================

from pathlib import Path
import argparse

import torch
from torch_geometric.loader import DataLoader

from train_gnn import (
    create_model,
    dataframe_to_graphs,
    load_split_dataframe,
    normalize_graph_targets,
    evaluate,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "gnn_experiments"
    / "gcn_scaffold"
    / "best_model.pt"
)

DEFAULT_TEST_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "aqsoldb_scaffold_test.csv"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved GCN/GAT checkpoint on an AqSolDB test set."
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Path to the saved .pt checkpoint.",
    )

    parser.add_argument(
        "--test-csv",
        type=Path,
        default=DEFAULT_TEST_CSV,
        help="Path to the test CSV.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Evaluation batch size.",
    )

    return parser.parse_args()


def choose_device(model_name: str) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    # GAT scatter operations may not be supported on Apple MPS.
    if model_name == "gat":
        return torch.device("cpu")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


def main() -> None:
    args = parse_arguments()

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{args.checkpoint}"
        )

    if not args.test_csv.exists():
        raise FileNotFoundError(
            f"Test CSV not found:\n{args.test_csv}"
        )

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    required_keys = {
        "model_name",
        "split_name",
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
    split_name = checkpoint["split_name"]
    target_mean = float(checkpoint["target_mean"])
    target_std = float(checkpoint["target_std"])

    device = choose_device(model_name)

    print("\nCheckpoint")
    print("=" * 70)
    print("Path:", args.checkpoint)
    print("Model:", model_name.upper())
    print("Split:", split_name)
    print("Best epoch:", checkpoint.get("epoch", "unknown"))
    print(
        "Saved validation RMSE:",
        checkpoint.get("validation_rmse", "unknown"),
    )
    print("Device:", device)

    print("\nArchitecture")
    print("=" * 70)
    print("Node feature dimension:", checkpoint["node_feature_dim"])
    print("Hidden dimension:", checkpoint["hidden_dim"])
    print("Number of layers:", checkpoint["num_layers"])
    print("Dropout:", checkpoint["dropout"])

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

    test_df = load_split_dataframe(
        args.test_csv,
        "test dataset",
    )

    test_graphs = dataframe_to_graphs(
        test_df,
        "test",
    )

    normalize_graph_targets(
        test_graphs,
        target_mean,
        target_std,
    )

    test_loader = DataLoader(
        test_graphs,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    (
        test_metrics,
        _,
        _,
        _,
    ) = evaluate(
        model=model,
        loader=test_loader,
        device=device,
        target_mean=target_mean,
        target_std=target_std,
    )

    print("\nTest results")
    print("=" * 70)
    print(f"RMSE: {test_metrics['rmse']:.4f}")
    print(f"MAE : {test_metrics['mae']:.4f}")
    print(f"R2  : {test_metrics['r2']:.4f}")

    print("\nEvaluation completed successfully.")


if __name__ == "__main__":
    main()
