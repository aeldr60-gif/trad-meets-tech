from __future__ import annotations
import pandas as pd


FEATURE_COLS = [
    "slot_position",
    "slots_per_measure",
    "notes_A","notes_B","notes_C","notes_D","notes_E","notes_F","notes_G",
    "rests",
]

def make_training_frames(beat_slots: pd.DataFrame):
    df = beat_slots.copy()

    for c in FEATURE_COLS:
        if c not in df.columns:
            df[c] = 0

    X = df[FEATURE_COLS].fillna(0)
    y_place = df["has_chord_here"].fillna(0).astype(int)

    tone_mask = y_place == 1
    X_tone = X.loc[tone_mask].copy()
    y_tone = df.loc[tone_mask, "chord_label"].fillna("").astype(str)

    return X, y_place, X_tone, y_tone
