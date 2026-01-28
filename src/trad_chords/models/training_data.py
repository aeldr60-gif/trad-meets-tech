from __future__ import annotations

from typing import Iterable, Optional, Tuple

import pandas as pd

from trad_chords.features.beat_slots import DEGREE_COLS
from trad_chords.models.feature_sets import get_feature_cols

"""
Helpers for preparing beat slot data for training the baseline chord prediction models.

This module standardizes chord presence labels (normalizing older
`chord_present` columns to `has_chord_here`) and provides
`make_training_frames()`, which extracts feature matrices and target vectors for
both placement and tone models. It validates that all required feature and label
columns are present, normalizes numeric and categorical dtypes, and returns
(X, y_place, X_tone, y_tone) in the exact format expected by the baseline
training code.
"""


def ensure_chord_present(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the chord-present label column.

    The project historically used both `has_chord_here` and `chord_present`.
    For consistency, we standardize on `has_chord_here`.
    """

    if "has_chord_here" in df.columns:
        return df

    if "chord_present" in df.columns:
        df = df.copy()
        df["has_chord_here"] = df["chord_present"].astype(int)
        return df

    raise ValueError("beat_slots is missing required column: has_chord_here (or chord_present)")


def make_training_frames(
    df: pd.DataFrame,
    *,
    feature_set: Optional[str] = None,
    feature_cols: Optional[Iterable[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Create X/y for the baseline models.

    Inputs expect a beat-slots DataFrame produced by `build_beat_slots`.

    - Placement model: predict whether a chord occurs on the slot (0/1).
    - Tone model: among true chord slots, predict a key-agnostic chord label
      (`chord_nashville`, like "deg_1:maj" or "deg_5:min").

    Features:
    - Numeric: part, slot_position, slots_per_measure, degree bins, rests
    - Categorical: type (jig/reel), music_mode (Major/Dorian/...)
    """

    df = ensure_chord_present(df)

    feature_cols_list = list(feature_cols) if feature_cols is not None else get_feature_cols(feature_set)

    required = {
        "has_chord_here",
        "chord_nashville",
        *feature_cols_list,
    }
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"beat_slots is missing required columns: {missing}")

    X = df[feature_cols_list].copy()

    # Normalize dtypes to avoid mixed-type warnings and sklearn failures.
    numeric_candidates = {
        "part",
        "slot_position",
        "slots_per_measure",
        "rests",
        *DEGREE_COLS,
    }
    categorical_candidates = {"type", "music_mode"}

    for c in set(X.columns) & numeric_candidates:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)
    for c in set(X.columns) & categorical_candidates:
        X[c] = X[c].astype(str).fillna("")
    y_place = df["has_chord_here"].astype(int)

    tone_mask = y_place == 1
    X_tone = X.loc[tone_mask].copy()
    y_tone = df.loc[tone_mask, "chord_nashville"].astype(str)

    return X, y_place, X_tone, y_tone
