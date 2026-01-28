from __future__ import annotations

"""
Centralized definitions of feature sets used by the baseline chord prediction models.

This module provides a single place to configure *which beat slot features* are fed
into the model pipelines, allowing experimentation with different subsets of the
existing feature engineering without modifying upstream code. Feature sets group
together structural, melodic, rhythmic, and metadata columns in various
combinations (e.g., all features, melody only, no metadata, rhythm only).

A feature set can be selected via the environment variable:

    TRAD_CHORDS_FEATURE_SET=all

or programmatically with `get_feature_cols("all")`. Unknown names raise a clear
error, ensuring reproducible and explicit model configurations.
"""


import os
from typing import Dict, List

from trad_chords.features.beat_slots import DEGREE_COLS


FEATURE_SETS: Dict[str, List[str]] = {
    # Baseline: everything we currently consider useful.
    "all": [
        "part",
        "slot_position",
        "slots_per_measure",
        *DEGREE_COLS,
        "rests",
        "type",
        "music_mode",
    ],
    "no_rests": [
        "part",
        "slot_position",
        "slots_per_measure",
        *DEGREE_COLS,
        "type",
        "music_mode",
    ],
    "include_key": [
        "part",
        "key",
        "music_mode"
        "slot_position",
        "slots_per_measure",
        *DEGREE_COLS,
        "rests",
        "type",
    ],
    # Drop potentially noisy metadata.
    "no_mode": [
        "part",
        "slot_position",
        "slots_per_measure",
        *DEGREE_COLS,
        "rests",
        "type",
    ],
    "no_meta": [
        "part",
        "slot_position",
        "slots_per_measure",
        *DEGREE_COLS,
    ],
    # Melody histograms only.
    "melody_only": [
        *DEGREE_COLS,
    ],
    "no_parts": [
        "slot_position",
        "slots_per_measure",
        *DEGREE_COLS,
        "type",
        "music_mode",
    ],
    # Melody + rhythmic position.
    "melody_plus_pos": [
        "slot_position",
        "slots_per_measure",
        *DEGREE_COLS,
    ],
    # Melody + meta (try to predict harmony from pitch content + mode/type, not position)
    "melody_plus_meta": [
        *DEGREE_COLS,
        "rests",
        "type",
        "music_mode",
    ],
    # Structure only.
    "pos_only": [
        "part",
        "slot_position",
        "slots_per_measure",
        "type",
        "music_mode",
    ],
}


def get_feature_cols(feature_set: str | None = None) -> List[str]:
    """Return feature columns for the requested set.

    If `feature_set` is omitted, reads TRAD_CHORDS_FEATURE_SET from the environment,
    defaulting to "all".
    """

    fs = feature_set or os.getenv("TRAD_CHORDS_FEATURE_SET", "all")
    if fs not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set '{fs}'. Valid: {sorted(FEATURE_SETS)}")
    return FEATURE_SETS[fs]
