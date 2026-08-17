"""Data loading, splitting, and leakage-safe preprocessing."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, StandardScaler

from .config import COMBINATIONS, PAPER_FEATURES, TARGET_COLUMN


@dataclass
class Dataset:
    features: pd.DataFrame
    target: pd.Series


@dataclass
class DataSplit:
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    train_indices: np.ndarray
    test_indices: np.ndarray


class TargetStandardizer:
    def __init__(self) -> None:
        self.mean = 0.0
        self.scale = 1.0

    def fit_transform(self, target: np.ndarray) -> np.ndarray:
        values = target.astype(np.float32).reshape(-1, 1)
        self.mean = float(values.mean())
        standard_deviation = float(values.std())
        self.scale = standard_deviation if standard_deviation > 1e-8 else 1.0
        return ((values - self.mean) / self.scale).astype(np.float32)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return (values.reshape(-1, 1) * self.scale + self.mean).reshape(-1)


def clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy().dropna(axis=0, how="all")
    cleaned.columns = [str(column).strip() for column in cleaned.columns]
    for column in cleaned.columns:
        if cleaned[column].dtype == object:
            cleaned[column] = cleaned[column].map(
                lambda value: value.strip() if isinstance(value, str) else value
            )
            cleaned[column] = cleaned[column].replace("", np.nan)
        numeric = pd.to_numeric(cleaned[column], errors="coerce")
        non_null = cleaned[column].notna()
        if non_null.any() and numeric[non_null].notna().mean() >= 0.9:
            cleaned[column] = numeric
    return cleaned


def load_datasets(data_path: Path) -> dict[str, Dataset]:
    workbook = pd.ExcelFile(data_path, engine="openpyxl")
    if len(workbook.sheet_names) < len(COMBINATIONS):
        raise ValueError("The workbook must contain at least three worksheets.")

    datasets: dict[str, Dataset] = {}
    for combination, sheet_name in zip(COMBINATIONS, workbook.sheet_names[:3]):
        frame = pd.read_excel(
            data_path,
            sheet_name=sheet_name,
            header=1,
            engine="openpyxl",
        )
        frame = clean_frame(frame)
        required = PAPER_FEATURES[combination] + [TARGET_COLUMN]
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing columns in {sheet_name!r}: {missing}")

        selected = frame[required].copy()
        selected[TARGET_COLUMN] = pd.to_numeric(
            selected[TARGET_COLUMN], errors="coerce"
        )
        selected = selected.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
        datasets[combination] = Dataset(
            features=selected[PAPER_FEATURES[combination]].copy(),
            target=selected[TARGET_COLUMN].astype(float),
        )
    return datasets


def split_dataset(dataset: Dataset, test_size: float, random_state: int) -> DataSplit:
    indices = np.arange(len(dataset.features))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )
    return DataSplit(
        x_train=dataset.features.iloc[train_indices].reset_index(drop=True),
        x_test=dataset.features.iloc[test_indices].reset_index(drop=True),
        y_train=dataset.target.iloc[train_indices].reset_index(drop=True),
        y_test=dataset.target.iloc[test_indices].reset_index(drop=True),
        train_indices=train_indices.copy(),
        test_indices=test_indices.copy(),
    )


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def feature_types(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = frame.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [column for column in frame.columns if column not in numeric]
    return numeric, categorical


def build_preprocessor(frame: pd.DataFrame, scale_numeric: bool) -> ColumnTransformer:
    numeric, categorical = feature_types(frame)
    transformers: list[tuple[str, Any, list[str]]] = []

    if numeric:
        numeric_transformer: Any = StandardScaler() if scale_numeric else "passthrough"
        transformers.append(("numeric", numeric_transformer, numeric))

    if categorical:
        transformers.append(("categorical", make_one_hot_encoder(), categorical))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_mlp_preprocessor(frame: pd.DataFrame, random_state: int) -> ColumnTransformer:
    numeric, categorical = feature_types(frame)
    transformers: list[tuple[str, Any, list[str]]] = []

    if numeric:
        numeric_pipeline = Pipeline(
            [
                (
                    "quantile",
                    QuantileTransformer(
                        n_quantiles=min(128, max(1, len(frame))),
                        output_distribution="normal",
                        random_state=random_state,
                    ),
                ),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric))

    if categorical:
        transformers.append(("categorical", make_one_hot_encoder(), categorical))

    return ColumnTransformer(transformers=transformers, remainder="drop")
