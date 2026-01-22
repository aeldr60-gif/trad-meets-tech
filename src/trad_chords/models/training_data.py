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

def ensure_chord_present(df):
    if "chord_present" in df.columns:
        return df

    # Prefer the “most normalized” chord field if it exists
    candidate_cols = [
        "chord_nashville",
        "chord_degree",
        "chord_degree_float",
        "chord_root_degree",
        "chord_label",
        "active_chord",
        "has_chord_here"
    ]
    for c in candidate_cols:
        if c in df.columns:
            s = df[c]
            df = df.copy()
            df["chord_present"] = (~s.isna()) & (s.astype(str).str.strip() != "") & (s.astype(str) != "None")
            return df

    raise ValueError(
        "beat_slots is missing required columns: ['chord_present'] and no chord column "
        f"was found among {candidate_cols}. Available columns: {list(df.columns)}"
    )



def make_training_frames(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Create X/y for the baseline models.

    Inputs expect a beat-slots DataFrame produced by `build_beat_slots`.

    - Placement model: predict whether a chord occurs on the slot (0/1).
    - Tone model: among true chord slots, predict a *key-agnostic* chord label
      (`chord_nashville`, like "deg_1:maj" or "deg_5:min").
    """

    df = ensure_chord_present(df)

    required = {"slot_position", "slots_per_measure", "has_chord_here", "chord_nashville", *DEGREE_COLS}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"beat_slots is missing required columns: {missing}")

    # These are the columns being trained on - this is what is most important for the model!!!
    feature_cols = ["part","slot_position", "slots_per_measure", *DEGREE_COLS]
    X = df[feature_cols].copy()
    y_place = df["has_chord_here"].astype(int)

    tone_mask = y_place == 1
    X_tone = X.loc[tone_mask].copy()
    y_tone = df.loc[tone_mask, "chord_nashville"].astype(str)

    return X, y_place, X_tone, y_tone
