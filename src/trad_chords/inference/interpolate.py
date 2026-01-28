from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from trad_chords.abc.render import build_chord_inserts_from_predictions, build_full_abc, render_abc_body_from_notes
from trad_chords.models.baseline import BaselineModels

"""
Inference utilities for inserting predicted chords into chordless ABC settings.

This module wraps the baseline placement/tone models and provides helpers for
ensuring required feature columns, running predictions on beat slot features, and
reconstructing ABC bodies with chord inserts. It aligns model predictions to
event-level ABC tokens using `build_chord_inserts_from_predictions()` and
re renders a clean ABC body and full ABC header via the render helpers.

The main entry point, `interpolate_chordless_to_csv()`, filters to chordless
settings, predicts chord presence and Nashville labels, converts them into chord
symbols, inserts them at the correct beat slots, and writes an output CSV with
both the ABC body and full ABC text for each tune.
"""


def _expected_columns_from_pipeline(pipe) -> Tuple[List[str], List[str]]:
    """Extract numeric + categorical column lists from a pipeline's ColumnTransformer."""
    pre = pipe.named_steps.get("pre")
    num_cols: List[str] = []
    cat_cols: List[str] = []
    if pre is None:
        return num_cols, cat_cols

    # transformers_ is populated after fitting
    for name, _transformer, cols in getattr(pre, "transformers_", []):
        if name == "num":
            num_cols.extend(list(cols))
        elif name == "cat":
            cat_cols.extend(list(cols))
    return num_cols, cat_cols


def ensure_columns_for_model(df: pd.DataFrame, models: BaselineModels) -> pd.DataFrame:
    """Add missing feature columns expected by the trained model pipelines."""
    df = df.copy()
    num_cols, cat_cols = _expected_columns_from_pipeline(models.placement)

    for c in num_cols:
        if c not in df.columns:
            df[c] = 0
    for c in cat_cols:
        if c not in df.columns:
            df[c] = "__MISSING__"

    return df


def predict_beat_slots(df_slots: pd.DataFrame, models: BaselineModels) -> pd.DataFrame:
    """Predict chord placement and tone on a beat_slots DataFrame."""
    df = ensure_columns_for_model(df_slots, models)

    pred_place = models.placement.predict(df)
    out = df_slots.copy()
    out["pred_chord_present"] = pred_place.astype(int)
    out["pred_chord_nashville"] = None

    mask = out["pred_chord_present"] == 1
    if int(mask.sum()) > 0:
        # tone model expects same columns (ensure again)
        df_tone = ensure_columns_for_model(df_slots.loc[mask].copy(), models)
        out.loc[mask, "pred_chord_nashville"] = models.tone.predict(df_tone)

    return out


def interpolate_chordless_to_csv(
    *,
    notes_table: pd.DataFrame,
    beat_slots: pd.DataFrame,
    chordless_index: pd.DataFrame,
    models: BaselineModels,
    out_csv: Path,
) -> pd.DataFrame:
    """Create an 'interpolated_tunes.csv' for chordless tunes.

    Output columns:
      setting_id, tune_id, name, type, meter, mode, abc_with_chords, full_abc
    """
    chordless_setting_ids = set(chordless_index["setting_id"].astype(int).tolist())

    notes_sub = notes_table[notes_table["setting_id"].astype(int).isin(chordless_setting_ids)].copy()
    beat_sub = beat_slots[beat_slots["setting_id"].astype(int).isin(chordless_setting_ids)].copy()

    beat_pred = predict_beat_slots(beat_sub, models)

    out_rows: List[dict] = []

    # index metadata lookup
    idx_meta = chordless_index.set_index("setting_id")

    for setting_id, ndf in notes_sub.groupby("setting_id", sort=False):
        sid = int(setting_id)
        meta = idx_meta.loc[sid]

        # limit to this tune's beat slots predictions
        bdf = beat_pred[beat_pred["setting_id"].astype(int) == sid].copy()

        inserts = build_chord_inserts_from_predictions(bdf, ndf)

        body = render_abc_body_from_notes(ndf, chord_inserts_by_event=inserts)
        full = build_full_abc(
            name=str(meta.get("name", "")),
            tune_type=str(meta.get("type", "")),
            meter=str(meta.get("meter", "")),
            mode_str=str(meta.get("mode", "")),
            abc_body=body,
        )

        out_rows.append(
            {
                "setting_id": sid,
                "tune_id": int(meta.get("tune_id", sid)),
                "name": str(meta.get("name", "")),
                "type": str(meta.get("type", "")),
                "meter": str(meta.get("meter", "")),
                "mode": str(meta.get("mode", "")),
                "abc_with_chords": body,
                "full_abc": full,
            }
        )

    out_df = pd.DataFrame(out_rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    return out_df
