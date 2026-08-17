"""Training-set-only randomized hyperparameter search."""

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold, ParameterSampler, RandomizedSearchCV

from .config import (
    BASELINE_SETTINGS,
    CLASSICAL_SEARCH_SPACES,
    MLP_SEARCH_SPACE,
)
from .data import DataSplit
from .models import (
    classical_pipeline,
    mlp_parameter_count,
    predict_mlp,
    regression_metrics,
    train_mlp,
)


@dataclass
class SearchResult:
    settings: dict[str, Any]
    mean_cv_r2: float
    std_cv_r2: float


def remove_model_prefix(parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        key.removeprefix("model__"): value for key, value in parameters.items()
    }


def classical_complexity(model_name: str, settings: dict[str, Any]) -> float:
    if model_name in {"RF", "ETR"}:
        depth = settings.get("max_depth")
        effective_depth = 50.0 if depth is None else float(depth)
        return (
            float(settings.get("n_estimators", 100))
            * effective_depth
            / max(1.0, float(settings.get("min_samples_leaf", 1)))
        )
    return float(settings.get("C", 1.0)) / max(
        0.01, float(settings.get("epsilon", 0.1))
    )


def search_space_size(search_space: Any) -> int:
    spaces = search_space if isinstance(search_space, list) else [search_space]
    return int(
        sum(
            math.prod(len(values) for values in conditional_space.values())
            for conditional_space in spaces
        )
    )


def search_classical_model(
    model_name: str,
    split: DataSplit,
    cross_validation: KFold,
    random_state: int,
    iterations: int,
    jobs: int,
) -> SearchResult:
    search_space = CLASSICAL_SEARCH_SPACES[model_name]
    total_combinations = search_space_size(search_space)
    search = RandomizedSearchCV(
        estimator=classical_pipeline(
            model_name,
            split.x_train,
            BASELINE_SETTINGS[model_name],
            random_state,
        ),
        param_distributions=search_space,
        n_iter=min(iterations, total_combinations),
        scoring={"r2": "r2", "neg_rmse": "neg_root_mean_squared_error"},
        refit=False,
        cv=cross_validation,
        n_jobs=jobs,
        error_score="raise",
    )
    search.fit(split.x_train, split.y_train)

    results = pd.DataFrame(search.cv_results_)
    candidate_settings = results["params"].map(remove_model_prefix)
    results["complexity"] = [
        classical_complexity(model_name, settings)
        for settings in candidate_settings
    ]
    best_r2 = float(results["mean_test_r2"].max())
    eligible = results[results["mean_test_r2"] >= best_r2 - 0.001]
    selected_index = eligible.sort_values(
        ["mean_test_neg_rmse", "complexity"],
        ascending=[False, True],
    ).index[0]

    return SearchResult(
        settings=remove_model_prefix(results.loc[selected_index, "params"]),
        mean_cv_r2=float(results.loc[selected_index, "mean_test_r2"]),
        std_cv_r2=float(results.loc[selected_index, "std_test_r2"]),
    )


def search_mlp(
    split: DataSplit,
    cross_validation: KFold,
    random_state: int,
    iterations: int,
    device: torch.device,
) -> SearchResult:
    total_combinations = math.prod(
        len(values) for values in MLP_SEARCH_SPACE.values()
    )
    sample_size = min(iterations, total_combinations)
    if sample_size < 1:
        raise ValueError("The number of MLP search iterations must be positive.")

    candidates = list(
        ParameterSampler(
            MLP_SEARCH_SPACE,
            n_iter=sample_size,
        )
    )
    candidate_results: list[dict[str, Any]] = []

    for settings in candidates:
        settings = {**settings, "hidden_layers": list(settings["hidden_layers"])}
        fold_scores: list[dict[str, float]] = []
        input_dimension = 0

        for fold_number, (train_index, validation_index) in enumerate(
            cross_validation.split(split.x_train), start=1
        ):
            fold_x_train = split.x_train.iloc[train_index].reset_index(drop=True)
            fold_y_train = split.y_train.iloc[train_index].reset_index(drop=True)
            fold_x_validation = split.x_train.iloc[validation_index].reset_index(
                drop=True
            )
            fold_y_validation = split.y_train.iloc[validation_index].to_numpy(
                dtype=float
            )

            model, preprocessor, target_transformer, input_dimension = train_mlp(
                fold_x_train,
                fold_y_train,
                settings,
                random_state + fold_number,
                device,
            )
            prediction = predict_mlp(
                model,
                preprocessor,
                target_transformer,
                fold_x_validation,
                device,
            )
            fold_scores.append(regression_metrics(fold_y_validation, prediction))
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        mean_r2 = float(np.mean([score["R2"] for score in fold_scores]))
        std_r2 = float(np.std([score["R2"] for score in fold_scores]))
        mean_rmse = float(np.mean([score["RMSE"] for score in fold_scores]))
        parameter_count = mlp_parameter_count(
            input_dimension, list(settings["hidden_layers"])
        )
        candidate_results.append(
            {
                "settings": settings,
                "mean_r2": mean_r2,
                "std_r2": std_r2,
                "mean_rmse": mean_rmse,
                "parameter_count": parameter_count,
            }
        )

    best_r2 = max(result["mean_r2"] for result in candidate_results)
    eligible = [
        result
        for result in candidate_results
        if result["mean_r2"] >= best_r2 - 0.001
    ]
    selected = min(
        eligible,
        key=lambda result: (result["mean_rmse"], result["parameter_count"]),
    )
    return SearchResult(
        settings=selected["settings"],
        mean_cv_r2=selected["mean_r2"],
        std_cv_r2=selected["std_r2"],
    )
