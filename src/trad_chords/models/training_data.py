from __future__ import annotations

from typing import Tuple

import pandas as pd


# Key-agnostic note representation: scale-degree bins.
# We include diatonic degrees 1..7 plus "in-between" chromatic bins (e.g., deg_1.5)
# encoded as deg_1_5.
DEGREE_COLS = [
    "deg_1",
    "deg_1_5",
    "deg_2",
    "deg_2_5",
    "deg_3",
    "deg_3_5",
    "deg_4",
    "deg_4_5",
    "deg_5",
    "deg_5_5",
    "deg_6",
    "deg_6_5",
    "deg_7",
    "deg_7_5",
]


def make_training_frames(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Create X/y for the baseline models.

    Inputs expect a beat-slots DataFrame produced by `build_beat_slots`.

    - Placement model: predict whether a chord occurs on the slot (0/1).
    - Tone model: among true chord slots, predict a *key-agnostic* chord label
      (`chord_nashville`, like "deg_1:maj" or "deg_5:min").
    """

    required = {"slot_position", "slots_per_measure", "chord_present", "chord_nashville", *DEGREE_COLS}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"beat_slots is missing required columns: {missing}")

    feature_cols = ["slot_position", "slots_per_measure", *DEGREE_COLS]
    X = df[feature_cols].copy()
    y_place = df["chord_present"].astype(int)

    tone_mask = y_place == 1
    X_tone = X.loc[tone_mask].copy()
    y_tone = df.loc[tone_mask, "chord_nashville"].astype(str)

    return X, y_place, X_tone, y_tone
