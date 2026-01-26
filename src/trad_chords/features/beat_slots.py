from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional

import re

import pandas as pd

from trad_chords.music.theory import (
    split_key_and_mode,
    parse_mode,
    scale_pitch_classes,
    abc_note_to_pitch_class,
    degree_bin_label,
    chord_symbol_to_pitch_class,
    chord_to_nashville,
)


DEGREE_COLS = [
    "deg_1", "deg_1_5", "deg_2", "deg_2_5", "deg_3", "deg_3_5", "deg_4", "deg_4_5",
    "deg_5", "deg_5_5", "deg_6", "deg_6_5", "deg_7", "deg_7_5",
]


def meter_to_slots(meter: str) -> int:
    """Map an ABC meter string like '4/4' or '6/8' to an integer slot count.

    We keep this simple and use the numerator as the number of slots.
    """
    num, _den = parse_meter(meter)
    return max(1, num)


def parse_meter(meter: str) -> Tuple[int, int]:
    """Parse a meter string like '4/4' or '6/8' into (numerator, denominator)."""
    try:
        s = str(meter).strip()
        if "/" in s:
            a, b = s.split("/", 1)
            num = int(a)
            den = int(b)
            return max(1, num), max(1, den)
    except Exception:
        pass
    return 4, 4


def default_abc_unit_length_whole(meter: str) -> float:
    """Return the default ABC unit note length (as a fraction of a whole note).

    ABC's default unit note length depends on the meter when no L: field is present:
    - If meter < 3/4, default is 1/16
    - Otherwise, default is 1/8

    See ABC 2.1 spec (common rule used by many renderers).
    """
    num, den = parse_meter(meter)
    try:
        ratio = float(num) / float(den)
        return 1.0 / 16.0 if ratio < 0.75 else 1.0 / 8.0
    except Exception:
        return 1.0 / 8.0


_L_FIELD_RE = re.compile(r"(?:\[)?L\s*:\s*(\d+)\s*/\s*(\d+)(?:\])?", re.IGNORECASE)
_LEN_SUFFIX_RE = re.compile(r"([0-9/]+)$")


def extract_l_field_whole(field_token_text: str) -> Optional[float]:
    """Extract L:1/8 from a token like '[L:1/8]' or 'L:1/8'."""
    m = _L_FIELD_RE.search(str(field_token_text))
    if not m:
        return None
    try:
        a = int(m.group(1))
        b = int(m.group(2))
        if a > 0 and b > 0:
            return float(a) / float(b)
    except Exception:
        return None
    return None


def length_multiplier_from_suffix(suffix: str) -> float:
    """Convert an ABC length suffix to a multiplier relative to the unit note length.

    Supported suffixes (common in TheSession bodies):
      ''   -> 1
      '2'  -> 2
      '/'  -> 1/2
      '/2' -> 1/2
      '3/2' -> 3/2
      '3/' -> 3/2
      '//' -> 1/4
      '3//' -> 3/4
    """
    s = str(suffix or "").strip()
    if not s:
        return 1.0
    if s.isdigit():
        return float(int(s))
    if s == "/":
        return 0.5
    if s == "//":
        return 0.25
    if s.startswith("/") and s[1:].isdigit():
        d = int(s[1:])
        return 1.0 / float(d) if d else 1.0
    if s.endswith("//") and s[:-2].isdigit():
        n = int(s[:-2])
        return float(n) / 4.0
    if s.endswith("/") and s[:-1].isdigit():
        n = int(s[:-1])
        return float(n) / 2.0
    if "/" in s:
        a, b = s.split("/", 1)
        if a.isdigit() and b.isdigit():
            bb = int(b)
            return float(int(a)) / float(bb) if bb else 1.0
    return 1.0


def token_duration_beats(token_text: str, meter: str, unit_len_whole: float) -> float:
    """Convert a note/rest token into duration in *meter denominator beats*.

    We interpret the meter as N/D (e.g., 4/4). One beat slot corresponds to 1/D of a whole note.
    Duration in beats = duration_whole / (1/D) = duration_whole * D.
    """
    _num, den = parse_meter(meter)
    m = _LEN_SUFFIX_RE.search(str(token_text))
    suffix = m.group(1) if m else ""
    mult = length_multiplier_from_suffix(suffix)
    dur_whole = float(unit_len_whole) * float(mult)
    return max(0.0, dur_whole * float(den))


def _new_slot(meta: Dict, slot_position: int, slots_per_measure: int, key: str, music_mode: str) -> Dict:
    row = {
        "tune_id": meta.get("tune_id"),
        "setting_id": meta.get("setting_id"),
        "name": meta.get("name"),
        "type": meta.get("type"),
        "meter": meta.get("meter"),
        "mode": meta.get("mode"),
        "tunebooks": meta.get("tunebooks"),
        "part": meta.get("part"),
        "measure_number": meta.get("measure_number"),
        "key": key,
        "music_mode": music_mode,
        "slot_position": slot_position,
        "slots_per_measure": slots_per_measure,
        "rests": 0,
        "has_chord_here": 0,
        "chord_label": None,
        "chord_root_degree": None,
        "chord_root_pc": None,
        "chord_quality": None,
        "chord_nashville": None,
    }
    for c in DEGREE_COLS:
        row[c] = 0
    return row


def build_beat_slots(notes_df: pd.DataFrame) -> pd.DataFrame:
    """Build a per-measure-slot feature table.

    Inputs:
      - notes_df: output of build_notes_table (tokenized ABC)

    Output:
      One row per (setting_id, part, measure_number, slot_position).
      Slots contain counts of notes in scale-degree bins + chord metadata.
    """

    required = {
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
    }
    missing = [c for c in required if c not in notes_df.columns]
    if missing:
        raise ValueError(f"notes_df is missing required columns: {missing}")

    # Determine the ABC unit note length (L:) for each setting.
    # If no L: is present, fall back to the ABC default based on meter.
    unit_len_whole_by_setting: Dict[int, float] = {}
    for sid, sg in notes_df.groupby(["setting_id"], sort=False):
        sid_i = int(sid)
        meter = str(sg["meter"].iloc[0] if len(sg) else "4/4")
        unit = default_abc_unit_length_whole(meter)

        # Scan field tokens in order; last L: wins.
        if "token_kind" in sg.columns and "token_text" in sg.columns:
            sg2 = sg.sort_values("event_index")
            for tk, tt in zip(sg2["token_kind"].astype(str), sg2["token_text"].astype(str)):
                if tk == "field":
                    maybe = extract_l_field_whole(tt)
                    if maybe is not None:
                        unit = maybe

        unit_len_whole_by_setting[sid_i] = float(unit)

    rows: List[Dict] = []

    grp_cols = ["setting_id", "part", "measure_number"]
    for (setting_id, part, measure_number), g in notes_df.groupby(grp_cols, sort=True):
        g = g.sort_values("event_index")
        meta = g.iloc[0].to_dict()

        mode_str = str(meta.get("mode") or "Cmajor")
        key_str, music_mode = split_key_and_mode(mode_str)
        tonic_pc, key_mode = parse_mode(mode_str)
        scale_pcs = scale_pitch_classes(tonic_pc, key_mode)

        meter = str(meta.get("meter") or "4/4")
        slots_per_measure = meter_to_slots(meter)
        unit_len_whole = unit_len_whole_by_setting.get(int(setting_id), default_abc_unit_length_whole(meter))

        # Create empty slots for this measure.
        slots: Dict[int, Dict] = {
            sp: _new_slot(meta, sp, slots_per_measure, key_str, music_mode) for sp in range(1, slots_per_measure + 1)
        }

        time_beats = 0.0
        for _, r in g.iterrows():
            kind = str(r["token_kind"])
            txt = str(r["token_text"])

            if kind == "bar":
                # Measure boundaries are already represented by measure_number; we don't allocate bars to slots.
                continue

            if kind in {"other", "ending", "field"}:
                # Non-time-bearing tokens should not shift the beat grid.
                continue

            # Slot position is the beat slot in which the token *starts*.
            slot_pos = int(time_beats) + 1
            if slot_pos < 1:
                slot_pos = 1
            if slot_pos > slots_per_measure:
                slot_pos = slots_per_measure

            slot = slots[slot_pos]

            if kind == "note":
                try:
                    pc = abc_note_to_pitch_class(txt, tonic_pc=tonic_pc, key_mode=key_mode)
                    deg_col = degree_bin_label(tonic_pc, scale_pcs, pc)
                    if deg_col in slot:
                        slot[deg_col] += 1
                except Exception:
                    # If we can't parse a note token, ignore it.
                    pass
                time_beats += token_duration_beats(txt, meter=meter, unit_len_whole=unit_len_whole)

            elif kind == "rest":
                slot["rests"] += 1
                time_beats += token_duration_beats(txt, meter=meter, unit_len_whole=unit_len_whole)

            elif kind == "chord":
                try:
                    root_pc, quality = chord_symbol_to_pitch_class(txt)
                    root_deg = degree_bin_label(tonic_pc, scale_pcs, root_pc)
                    slot["has_chord_here"] = 1
                    slot["chord_label"] = txt
                    slot["chord_root_degree"] = root_deg
                    slot["chord_root_pc"] = int(root_pc)
                    slot["chord_quality"] = quality
                    slot["chord_nashville"] = chord_to_nashville(tonic_pc, scale_pcs, root_pc, quality)
                except Exception:
                    pass

            # Chord tokens do not advance time.

        rows.extend([slots[sp] for sp in range(1, slots_per_measure + 1)])

    df = pd.DataFrame(rows)

    # Stable column order for downstream steps.
    first_cols = [
        "tune_id",
        "setting_id",
        "name",
        "type",
        "meter",
        "mode",
        "tunebooks",
        "part",
        "measure_number",
        "key",
        "music_mode",
        "slot_position",
        "slots_per_measure",
        *DEGREE_COLS,
        "rests",
        "has_chord_here",
        "chord_label",
        "chord_root_degree",
        "chord_root_pc",
        "chord_quality",
        "chord_nashville",
    ]
    return df[first_cols]


def write_beat_slots(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
