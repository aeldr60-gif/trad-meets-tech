from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class BaselineModels:
    placement: Pipeline
    tone: Pipeline

    def save(self, model_dir: Path) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.placement, model_dir / "placement.joblib")
        joblib.dump(self.tone, model_dir / "tone.joblib")

    @staticmethod
    def load(model_dir: Path) -> "BaselineModels":
        return BaselineModels(
            placement=joblib.load(model_dir / "placement.joblib"),
            tone=joblib.load(model_dir / "tone.joblib"),
        )


def train_baseline(X, y_place, X_tone, y_tone, seed: int = 42) -> BaselineModels:
    placement = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, random_state=seed)),
    ])
    placement.fit(X, y_place)

    tone = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
    ])
    if len(y_tone) == 0:
        tone.fit(X.iloc[:1], [""])
    else:
        tone.fit(X_tone, y_tone)

    return BaselineModels(placement=placement, tone=tone)
