"""Command-line orchestration for the complete analysis."""

import argparse
from pathlib import Path

import torch
from sklearn.model_selection import KFold

from .config import (
    BASELINE_SETTINGS,
    COMBINATIONS,
    DEFAULT_ZSCORE_THRESHOLD,
    MODELS,
)
from .data import load_datasets, split_dataset
from .models import evaluate_classical, evaluate_mlp
from .outliers import filter_training_outliers
from .search import SearchResult, search_classical_model, search_mlp


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(requested)


def print_metrics(
    combination: str,
    model_name: str,
    before: dict[str, float],
    after: dict[str, float],
) -> None:
    print(
        f"{combination},{model_name},"
        f"{before['R2']:.6f},{after['R2']:.6f},"
        f"{before['RMSE']:.6f},{after['RMSE']:.6f},"
        f"{before['MAE']:.6f},{after['MAE']:.6f}"
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run hyperparameter optimization for the SMX models."
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to the source Excel workbook.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=777)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument(
        "--zscore-threshold",
        type=float,
        default=DEFAULT_ZSCORE_THRESHOLD,
        help="Training-only numerical-feature outlier threshold.",
    )
    parser.add_argument("--classical-iterations", type=int, default=50)
    parser.add_argument("--mlp-iterations", type=int, default=50)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel jobs used by RandomizedSearchCV.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    device = resolve_device(arguments.device)
    datasets = load_datasets(arguments.data.resolve())
    splits = {
        combination: split_dataset(
            datasets[combination],
            arguments.test_size,
            arguments.random_state,
        )
        for combination in COMBINATIONS
    }
    splits, _ = filter_training_outliers(
        splits,
        threshold=arguments.zscore_threshold,
        reference_combination="C3",
    )

    cross_validation = KFold(
        n_splits=arguments.cv_folds,
        shuffle=True,
        random_state=arguments.random_state,
    )
    search_split = splits["C3"]
    selected: dict[str, SearchResult] = {}

    for model_name in ("RF", "SVR", "ETR"):
        selected[model_name] = search_classical_model(
            model_name,
            search_split,
            cross_validation,
            arguments.random_state,
            arguments.classical_iterations,
            arguments.n_jobs,
        )

    selected["MLP"] = search_mlp(
        search_split,
        cross_validation,
        arguments.random_state,
        arguments.mlp_iterations,
        device,
    )

    print(
        "Combination,Model,Before_R2,After_R2,Before_RMSE,After_RMSE,"
        "Before_MAE,After_MAE"
    )
    for combination in COMBINATIONS:
        split = splits[combination]
        for model_name in MODELS:
            if model_name == "MLP":
                before = evaluate_mlp(
                    split,
                    BASELINE_SETTINGS[model_name],
                    arguments.random_state,
                    device,
                )
                after = evaluate_mlp(
                    split,
                    selected[model_name].settings,
                    arguments.random_state,
                    device,
                )
            else:
                before = evaluate_classical(
                    model_name,
                    split,
                    BASELINE_SETTINGS[model_name],
                    arguments.random_state,
                )
                after = evaluate_classical(
                    model_name,
                    split,
                    selected[model_name].settings,
                    arguments.random_state,
                )
            print_metrics(combination, model_name, before, after)
