"""Training-only Z-score screening for numerical input variables."""

from dataclasses import dataclass

import numpy as np

from .config import COMBINATIONS
from .data import DataSplit


@dataclass
class OutlierReport:
    threshold: float
    reference_combination: str
    numerical_features: list[str]
    initial_training_observations: int
    removed_training_observations: int
    retained_training_observations: int
    untouched_test_observations: int
    flagged_by_feature: dict[str, int]


def filter_training_outliers(
    splits: dict[str, DataSplit],
    threshold: float,
    reference_combination: str = "C3",
) -> tuple[dict[str, DataSplit], OutlierReport]:
    if threshold <= 0:
        raise ValueError("The Z-score threshold must be positive.")
    if reference_combination not in splits:
        raise KeyError(f"Unknown reference combination: {reference_combination}")

    reference = splits[reference_combination]
    numerical = reference.x_train.select_dtypes(include=[np.number])
    if numerical.empty:
        raise ValueError("No numerical training features were found for Z-score screening.")

    means = numerical.mean(axis=0, skipna=True)
    standard_deviations = numerical.std(axis=0, ddof=0, skipna=True)
    valid_standard_deviations = standard_deviations.replace(0.0, np.nan)
    absolute_z_scores = ((numerical - means) / valid_standard_deviations).abs()
    flagged_cells = absolute_z_scores.gt(threshold).fillna(False)
    flagged_rows = flagged_cells.any(axis=1).to_numpy(dtype=bool)
    retained_rows = ~flagged_rows

    filtered: dict[str, DataSplit] = {}
    for combination in COMBINATIONS:
        split = splits[combination]
        if not np.array_equal(split.train_indices, reference.train_indices):
            raise ValueError("Training rows are not aligned across C1, C2, and C3.")
        if not np.array_equal(split.test_indices, reference.test_indices):
            raise ValueError("Test rows are not aligned across C1, C2, and C3.")

        filtered[combination] = DataSplit(
            x_train=split.x_train.loc[retained_rows].reset_index(drop=True),
            x_test=split.x_test.copy(),
            y_train=split.y_train.loc[retained_rows].reset_index(drop=True),
            y_test=split.y_test.copy(),
            train_indices=split.train_indices[retained_rows].copy(),
            test_indices=split.test_indices.copy(),
        )

    removed = int(flagged_rows.sum())
    report = OutlierReport(
        threshold=float(threshold),
        reference_combination=reference_combination,
        numerical_features=numerical.columns.tolist(),
        initial_training_observations=len(reference.x_train),
        removed_training_observations=removed,
        retained_training_observations=int(retained_rows.sum()),
        untouched_test_observations=len(reference.x_test),
        flagged_by_feature={
            column: int(flagged_cells[column].sum()) for column in numerical.columns
        },
    )
    return filtered, report
