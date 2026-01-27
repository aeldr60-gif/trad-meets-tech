from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import json
import sys

import numpy as np
import pandas as pd
import sklearn

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from trad_chords.features.beat_slots import DEGREE_COLS


@dataclass
class BaselineModels:
    placement: Pipeline
    tone: Pipeline

    def save(self, model_dir: Path) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.placement, model_dir / "placement.joblib")
        joblib.dump(self.tone, model_dir / "tone.joblib")

        # Write environment metadata alongside the model files to make loading issues
        # (version skew) easier to diagnose and to improve reproducibility.
        meta = {
            "python": sys.version.replace("\n", " "),
            "numpy": getattr(np, "__version__", ""),
            "pandas": getattr(pd, "__version__", ""),
            "sklearn": getattr(sklearn, "__version__", ""),
            "joblib": getattr(joblib, "__version__", ""),
        }
        (model_dir / "model_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @staticmethod
    def load(model_dir: Path) -> "BaselineModels":
        return BaselineModels(
            placement=joblib.load(model_dir / "placement.joblib"),
            tone=joblib.load(model_dir / "tone.joblib"),
        )


def _cols_present(X, cols: List[str]) -> List[str]:
    return [c for c in cols if c in getattr(X, "columns", [])]


def build_preprocessor(X) -> ColumnTransformer:
    """Build a numeric+categorical preprocessor for the given frame."""

    numeric_cols = _cols_present(
        X,
        [
            "part",
            "slot_position",
            "slots_per_measure",
            "rests",
            *DEGREE_COLS,
        ],
    )

    categorical_cols = _cols_present(X, ["type", "music_mode"])

    transformers = []
    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))
    if categorical_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols))

    if not transformers:
        raise ValueError("No usable feature columns found to build a preprocessor.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def train_baseline(X, y_place, X_tone, y_tone, seed: int = 42) -> BaselineModels:
    """Train the baseline placement + tone models."""

    pre_place = build_preprocessor(X)

    placement = Pipeline(
        steps=[
            ("pre", pre_place),
            ("clf", LogisticRegression(max_iter=2000, solver="saga", random_state=seed)),
        ]
    )

    tone = Pipeline(
        steps=[
            ("pre", build_preprocessor(X_tone) if len(X_tone) else pre_place),
            ("clf", LogisticRegression(max_iter=2000, solver="saga",  random_state=seed)),
        ]
    )

    placement.fit(X, y_place)

    if len(X_tone) > 0 and y_tone.nunique() > 1:
        tone.fit(X_tone, y_tone)

    return BaselineModels(placement=placement, tone=tone)
