from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from trad_chords.config import load_config
from trad_chords.features.beat_slots import DEGREE_COLS, parse_meter, default_abc_unit_length_whole, extract_l_field_whole, token_duration_beats
from trad_chords.music.theory import parse_mode, scale_pitch_classes, nashville_to_chord_symbol, mode_to_abc_key
from trad_chords.models.baseline import BaselineModels


FEATURE_COLS = [
    "part",
    "slot_position",
    "slots_per_measure",
    *DEGREE_COLS,
    "rests",
    "type",
    "music_mode",
]


def _meter_to_unit_length(meter: str) -> str:
    """Choose a reasonable default L: for full_abc.

    Many TheSession tunes use 6/8 (jigs) or 4/4 (reels). A common convention is:
    - 6/8 -> L:1/8
    - 4/4 -> L:1/8
    """
    # Keep it simple and consistent.
    return "1/8"


def _build_full_abc(name: str, meter: str, mode: str, body: str, ref: str) -> str:
    # Minimal header that most ABC players accept.
    # We use setting_id as X: so multiple settings won't collide.
    lines = [
        f"X:{ref}",
        f"T:{name}",
        f"M:{meter}",
        f"L:{_meter_to_unit_length(meter)}",
        f"K:{mode_to_abc_key(mode)}",
        body.strip(),
    ]
    return "\n".join(lines) + "\n"


def harmonize_chordless(config_path: str = "configs/default.yaml") -> Path:
    """Predict chords for chordless tunes and write a CSV with ABC strings.

    Output CSV columns include:
    - abc_with_pred_chords: body string with inserted chord annotations
    - full_abc: header + body suitable for copy/paste into ABC tools

    Returns the path to the written CSV.
    """
    cfg = load_config(config_path)

    chordless_index = pd.read_csv(cfg.artifacts.chordless_index_csv)
    if chordless_index.empty:
        raise ValueError("chordless_index is empty; run `trad-chords split-index` first.")

    # Load precomputed artifacts and filter to chordless settings.
    # This avoids recomputing large intermediate tables during inference.
    chordless_ids = set(chordless_index["setting_id"].astype(int).tolist())

    notes_usecols = [
        "tune_id",
        "setting_id",
        "name",
        "type",
        "meter",
        "mode",
        "tunebooks",
        "part",
        "measure_number",
        "event_index",
        "token_kind",
        "token_text",
    ]
    notes = pd.read_csv(cfg.artifacts.notes_table_csv, usecols=notes_usecols)
    notes = notes[notes["setting_id"].astype(int).isin(chordless_ids)].copy()

    beat_usecols = [
        "setting_id",
        "part",
        "measure_number",
        "slot_position",
        "slots_per_measure",
        "mode",
        "type",
        "music_mode",
        "rests",
        *DEGREE_COLS,
    ]
    beat = pd.read_csv(cfg.artifacts.beat_slots_csv, usecols=beat_usecols)
    beat = beat[beat["setting_id"].astype(int).isin(chordless_ids)].copy()

    # Load models
    models = BaselineModels.load(cfg.paths.model_dir)

    # Build X for inference
    missing = [c for c in FEATURE_COLS if c not in beat.columns]
    if missing:
        raise ValueError(
            f"beat_slots is missing required feature columns for inference: {missing}. "
            "Regenerate beat_slots with the updated pipeline."
        )

    X = beat[FEATURE_COLS].copy()

    # Predict placement
    place_pred = models.placement.predict(X)
    beat["pred_has_chord"] = place_pred.astype(int)

    # Predict tone only where a chord is predicted
    tone_mask = beat["pred_has_chord"] == 1
    beat["pred_chord_nashville"] = ""
    if int(tone_mask.sum()) > 0:
        beat.loc[tone_mask, "pred_chord_nashville"] = models.tone.predict(X.loc[tone_mask]).astype(str)

    # Convert Nashville label to chord symbol in the local key/mode
    cache: Dict[str, Tuple[int, object, Tuple[int, ...]]] = {}

    def to_symbol(mode_str: str, nash: str) -> str:
        if not nash:
            return ""
        if mode_str not in cache:
            tonic_pc, km = parse_mode(mode_str)
            scale_pcs = tuple(scale_pitch_classes(tonic_pc, km))
            cache[mode_str] = (tonic_pc, km, scale_pcs)
        tonic_pc, km, scale_pcs = cache[mode_str]
        # nashville_to_chord_symbol expects a KeyMode object, not a scale list.
        return nashville_to_chord_symbol(tonic_pc, km, nash)

    beat["pred_chord_symbol"] = [
        to_symbol(m, n) for m, n in zip(beat["mode"].astype(str), beat["pred_chord_nashville"].astype(str))
    ]

    # Build lookup for insertion: (setting_id, part, measure_number, slot_position) -> chord symbol
    pred_map: Dict[Tuple[int, int, int, int], str] = {}
    for r in beat.itertuples(index=False):
        if getattr(r, "pred_has_chord") == 1:
            key = (int(r.setting_id), int(r.part), int(r.measure_number), int(r.slot_position))
            sym = getattr(r, "pred_chord_symbol")
            if isinstance(sym, str) and sym:
                pred_map[key] = sym

    # slots_per_measure lookup per (setting_id, part, measure_number)
    spm_map: Dict[Tuple[int, int, int], int] = {}
    for (sid, part, meas), g in beat.groupby(["setting_id", "part", "measure_number"], sort=False):
        try:
            spm_map[(int(sid), int(part), int(meas))] = int(g["slots_per_measure"].iloc[0])
        except Exception:
            continue

    # Stitch notes back into ABC with inserted chord tokens
    out_rows = []
    for (setting_id,), sdf in notes.groupby(["setting_id"], sort=True):
        sdf = sdf.sort_values(["part", "measure_number", "event_index"], kind="mergesort")
        meta = sdf.iloc[0]
        name = str(meta.get("name", ""))
        meter = str(meta.get("meter", ""))
        mode = str(meta.get("mode", ""))
        tune_id = int(meta.get("tune_id", 0))
        tune_type = str(meta.get("type", ""))
        tunebooks = int(meta.get("tunebooks", 0))

        body_tokens = []
        current_part = None
        current_measure = None
        inserted = set()
        slots_per_measure = None

        # Slot-time tracking. We measure time in "meter denominator beats" (e.g., quarter-beats in 4/4).
        meter_for_setting = meter
        unit_len_whole = default_abc_unit_length_whole(meter_for_setting)
        time_beats = 0.0

        for row in sdf.itertuples(index=False):
            part = int(row.part)
            measure = int(row.measure_number)
            kind = str(row.token_kind)
            txt = str(row.token_text)

            # Reset per-measure time tracking
            if current_part != part or current_measure != measure:
                current_part = part
                current_measure = measure
                inserted = set()
                time_beats = 0.0
                slots_per_measure = spm_map.get((int(setting_id), part, measure), None)

            if kind == "bar":
                body_tokens.append(txt)
                continue

            # Update unit length if this setting's ABC specifies L:...
            if kind == "field":
                maybe = extract_l_field_whole(txt)
                if maybe is not None:
                    unit_len_whole = maybe
                body_tokens.append(txt)
                continue

            if kind in {"ending", "other"}:
                body_tokens.append(txt)
                continue

            # Compute the current slot based on start time
            spm = int(slots_per_measure) if slots_per_measure else None
            slot_pos = int(time_beats) + 1
            if spm is not None:
                if slot_pos < 1:
                    slot_pos = 1
                if slot_pos > spm:
                    slot_pos = spm

            # Insert predicted chord only immediately before time-bearing tokens.
            if kind in {"note", "rest"}:
                k = (int(setting_id), part, measure, int(slot_pos))
                if int(slot_pos) not in inserted:
                    sym = pred_map.get(k, "")
                    if sym:
                        body_tokens.append(f'"{sym}"')
                    inserted.add(int(slot_pos))

            # Re-emit original token
            if kind == "chord":
                # Rare in chordless tunes, but keep output valid if it happens.
                body_tokens.append(f'"{txt}"')
            else:
                body_tokens.append(txt)

            # Advance time only for note/rest, using their ABC length
            if kind in {"note", "rest"}:
                time_beats += token_duration_beats(txt, meter=meter_for_setting, unit_len_whole=unit_len_whole)

        abc_with_chords = "".join(body_tokens)
        full_abc = _build_full_abc(name=name, meter=meter, mode=mode, body=abc_with_chords, ref=str(setting_id))

        out_rows.append(
            {
                "tune_id": tune_id,
                "setting_id": int(setting_id),
                "name": name,
                "type": tune_type,
                "meter": meter,
                "mode": mode,
                "tunebooks": tunebooks,
                "abc_with_pred_chords": abc_with_chords,
                "full_abc": full_abc,
            }
        )

    out_df = pd.DataFrame(out_rows)

    out_dir = cfg.paths.outputs_dir / "interpolated"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "chordless_interpolated.csv"
    out_df.to_csv(out_path, index=False)

    return out_path
