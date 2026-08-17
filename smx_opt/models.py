"""Model definitions, training, prediction, and evaluation."""

import math
import random
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import MLP_TRAINING_EPOCHS
from .data import (
    DataSplit,
    TargetStandardizer,
    build_mlp_preprocessor,
    build_preprocessor,
)


class TabularMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: list[int], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.extend(
                [
                    nn.Linear(previous_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                ]
            )
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "R2": float(r2_score(target, prediction)),
        "RMSE": float(math.sqrt(mean_squared_error(target, prediction))),
        "MAE": float(mean_absolute_error(target, prediction)),
    }


def classical_pipeline(
    model_name: str,
    frame: pd.DataFrame,
    settings: dict[str, Any],
    random_state: int,
) -> Pipeline:
    if model_name == "RF":
        estimator = RandomForestRegressor(
            **settings,
            random_state=random_state,
            n_jobs=1,
        )
        scale_numeric = False
    elif model_name == "ETR":
        estimator = ExtraTreesRegressor(
            **settings,
            random_state=random_state,
            n_jobs=1,
        )
        scale_numeric = False
    elif model_name == "SVR":
        estimator = SVR(**settings)
        scale_numeric = True
    else:
        raise ValueError(f"Unsupported classical model: {model_name}")

    return Pipeline(
        [
            ("preprocessor", build_preprocessor(frame, scale_numeric)),
            ("model", estimator),
        ]
    )


def train_mlp(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    settings: dict[str, Any],
    random_state: int,
    device: torch.device,
) -> tuple[TabularMLP, ColumnTransformer, TargetStandardizer, int]:
    set_seed(random_state)
    preprocessor = build_mlp_preprocessor(x_train, random_state)
    transformed_x = preprocessor.fit_transform(x_train).astype(np.float32)
    target_transformer = TargetStandardizer()
    transformed_y = target_transformer.fit_transform(
        y_train.to_numpy(dtype=np.float32)
    )

    model = TabularMLP(
        input_dim=transformed_x.shape[1],
        hidden_layers=list(settings["hidden_layers"]),
        dropout=float(settings["dropout"]),
    ).to(device)
    dataset = TensorDataset(
        torch.from_numpy(transformed_x),
        torch.from_numpy(transformed_y),
    )
    generator = torch.Generator().manual_seed(random_state)
    loader = DataLoader(
        dataset,
        batch_size=int(settings["batch_size"]),
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["lr"]),
        weight_decay=float(settings["weight_decay"]),
    )
    loss_function = nn.MSELoss()

    for _ in range(MLP_TRAINING_EPOCHS):
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

    return model, preprocessor, target_transformer, int(transformed_x.shape[1])


def predict_mlp(
    model: TabularMLP,
    preprocessor: ColumnTransformer,
    target_transformer: TargetStandardizer,
    frame: pd.DataFrame,
    device: torch.device,
) -> np.ndarray:
    transformed = preprocessor.transform(frame).astype(np.float32)
    inputs = torch.from_numpy(transformed).to(device)
    model.eval()
    with torch.no_grad():
        scaled_prediction = model(inputs).cpu().numpy().reshape(-1)
    return target_transformer.inverse_transform(scaled_prediction)


def mlp_parameter_count(input_dim: int, hidden_layers: list[int]) -> int:
    total = 0
    previous_dim = input_dim
    for hidden_dim in hidden_layers:
        total += (previous_dim + 1) * hidden_dim
        total += 2 * hidden_dim
        previous_dim = hidden_dim
    return int(total + previous_dim + 1)


def evaluate_classical(
    model_name: str,
    split: DataSplit,
    settings: dict[str, Any],
    random_state: int,
) -> dict[str, float]:
    model = classical_pipeline(
        model_name,
        split.x_train,
        settings,
        random_state,
    )
    model.fit(split.x_train, split.y_train)
    prediction = model.predict(split.x_test)
    return regression_metrics(split.y_test.to_numpy(dtype=float), prediction)


def evaluate_mlp(
    split: DataSplit,
    settings: dict[str, Any],
    random_state: int,
    device: torch.device,
) -> dict[str, float]:
    model, preprocessor, target_transformer, _ = train_mlp(
        split.x_train,
        split.y_train,
        settings,
        random_state,
        device,
    )
    prediction = predict_mlp(
        model,
        preprocessor,
        target_transformer,
        split.x_test,
        device,
    )
    return regression_metrics(split.y_test.to_numpy(dtype=float), prediction)
