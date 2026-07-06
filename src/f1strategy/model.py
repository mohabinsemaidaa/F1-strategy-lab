"""Lap-time prediction model.

A gradient-boosted tree regressor inside a sklearn Pipeline, so
preprocessing (one-hot encoding) travels with the model artifact.

Validation uses GroupKFold grouped by Event: we always evaluate on races
the model has never seen, which is the honest way to measure this —
random splits would leak laps from the same race into train and test.
"""

from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

MODEL_PATH = Path("models/lap_time_model.joblib")

def make_pipeline() -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("num", "passthrough", NUMERIC_FEATURES),
        ]
    )
    model = HistGradientBoostingRegressor(
        max_iter=400,
        learning_rate=0.06,
        max_depth=None,
        min_samples_leaf=40,
        random_state=42,
    )
    return Pipeline([("preprocess", preprocess), ("model", model)])

def cross_validate(X: pd.DataFrame, y: pd.Series, n_splits: int = 4) -> dict:
    """Leave-races-out cross-validation. Returns MAE per fold and overall."""
    groups = X["Event"]
    n_splits = min(n_splits, groups.nunique())
    if n_splits < 2:
        return {"fold_mae": [], "mean_mae": float("nan")}

    gkf = GroupKFold(n_splits=n_splits)
    fold_mae = []
    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        pipe = make_pipeline()
        pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = pipe.predict(X.iloc[test_idx])
        fold_mae.append(float(mean_absolute_error(y.iloc[test_idx], pred)))
    return {"fold_mae": fold_mae, "mean_mae": float(np.mean(fold_mae))}

def train_final(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    """Fit on everything (used after CV has reported honest numbers)."""
    pipe = make_pipeline()
    pipe.fit(X, y)
    return pipe

def save_model(pipe: Pipeline, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, path)

def load_model(path: Path = MODEL_PATH) -> Pipeline:
    return joblib.load(path)