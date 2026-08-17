"""Model settings and the manuscript-defined feature combinations."""

from typing import Any


TARGET_COLUMN = "Adsorption capacity (mg/g)"
COMBINATIONS = ("C1", "C2", "C3")
MODELS = ("RF", "SVR", "ETR", "MLP")
DEFAULT_ZSCORE_THRESHOLD = 3.0
MLP_TRAINING_EPOCHS = 182

PAPER_FEATURES: dict[str, list[str]] = {
    "C1": [
        "Biomass",
        "Heating rate (℃/min)",
        "Pyrolysis temperature (℃)",
        "Pyrolysis time（h）",
        "Modification method",
        "pH",
        "Adsorption time (h)",
        "Adsorbent dosage (g/l)",
        "Adsorption temperature (℃)",
        "Initial concentration (mg/l)",
    ],
    "C2": [
        "C (%)",
        "H (%)",
        "O (%)",
        "N (%)",
        "Specific surface area (m2/g）",
        "Pore volume (cm3/g）",
        "Average pore size (nm)",
        "pH",
        "Adsorption time (h)",
        "Adsorbent dosage (g/l)",
        "Adsorption temperature (℃)",
        "Initial concentration (mg/l)",
    ],
    "C3": [
        "Biomass",
        "Heating rate (℃/min)",
        "Pyrolysis temperature (℃)",
        "Pyrolysis time（h）",
        "Modification method",
        "C (%)",
        "H (%)",
        "O (%)",
        "N (%)",
        "Specific surface area (m2/g）",
        "Pore volume (cm3/g）",
        "Average pore size (nm)",
        "pH",
        "Adsorption time (h)",
        "Adsorbent dosage (g/l)",
        "Adsorption temperature (℃)",
        "Initial concentration (mg/l)",
    ],
}

BASELINE_SETTINGS: dict[str, dict[str, Any]] = {
    "RF": {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": 1.0,
    },
    "SVR": {
        "kernel": "rbf",
        "C": 1.0,
        "gamma": "scale",
        "epsilon": 0.1,
    },
    "ETR": {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": 1.0,
    },
    "MLP": {
        "hidden_layers": [64, 32],
        "dropout": 0.0,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "batch_size": 32,
    },
}

CLASSICAL_SEARCH_SPACES: dict[str, Any] = {
    "RF": {
        "model__n_estimators": [200, 400, 600, 800],
        "model__max_depth": [None, 10, 20, 40],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", 0.5, 1.0],
    },
    "SVR": {
        "model__kernel": ["linear", "rbf", "poly"],
        "model__C": [0.1, 1.0, 10.0, 100.0, 300.0],
        "model__gamma": ["scale", "auto", 0.001, 0.01, 0.1],
        "model__epsilon": [0.01, 0.05, 0.1, 0.2, 0.5],
    },
    "ETR": {
        "model__n_estimators": [200, 400, 700, 900],
        "model__max_depth": [None, 10, 20, 40],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", 0.5, 1.0],
    },
}

MLP_SEARCH_SPACE: dict[str, list[Any]] = {
    "hidden_layers": [
        [32, 32],
        [64, 32],
        [128, 64],
        [256, 128, 64],
        [512, 256, 128, 64],
        [1024, 1024, 512, 256, 128, 64],
    ],
    "dropout": [0.0, 0.1, 0.2, 0.3, 0.5],
    "lr": [3e-4, 5e-4, 1e-3, 2e-3],
    "weight_decay": [0.0, 1e-5, 1e-4],
    "batch_size": [8, 16, 32],
}
