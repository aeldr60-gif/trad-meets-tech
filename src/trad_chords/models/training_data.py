from __future__ import annotations

from typing import Tuple

import pandas as pd

from trad_chords.features.beat_slots import DEGREE_COLS


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


def make_training_frames(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
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

    required = {
        "part",
        "slot_position",
        "slots_per_measure",
        "type",
        "music_mode",
        "has_chord_here",
        "chord_nashville",
        "rests",
        *DEGREE_COLS,
    }
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"beat_slots is missing required columns: {missing}")

    feature_cols = [
        "part",
        "slot_position",
        "slots_per_measure",
        *DEGREE_COLS,
        "rests",
        "type",
        "music_mode",
    ]

    X = df[feature_cols].copy()
    y_place = df["has_chord_here"].astype(int)

    tone_mask = y_place == 1
    X_tone = X.loc[tone_mask].copy()
    y_tone = df.loc[tone_mask, "chord_nashville"].astype(str)

    return X, y_place, X_tone, y_tone
