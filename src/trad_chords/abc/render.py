from __future__ import annotations

"""ABC rendering helpers.

This module is used by the (optional) `inference.interpolate` path.

The active CLI pipeline uses `inference.harmonize`, which directly inserts
chord tokens while iterating the notes table. This module keeps a more
general "render from notes + chord inserts" API.
"""

from collections import defaultdict
from typing import Dict, List, Mapping, Tuple

import pandas as pd

from trad_chords.features.beat_slots import meter_to_slots
from trad_chords.music.theory import mode_to_abc_key, nashville_to_chord_symbol, parse_mode


def render_abc_body_from_notes(
    notes_df: pd.DataFrame,
    chord_inserts_by_event: Mapping[int, List[str]] | None = None,
) -> str:
    """Render an ABC body string from a notes_table slice.

    `notes_df` must include: event_index, token_kind, token_text.
    `chord_inserts_by_event` maps event_index -> list of chord symbols to inject
    (as raw chord text, without quotes).
    """
    chord_inserts_by_event = chord_inserts_by_event or {}
    parts: List[str] = []

    for _, r in notes_df.sort_values("event_index").iterrows():
        ev_i = int(r["event_index"])
        for ch in chord_inserts_by_event.get(ev_i, []):
            if ch:
                parts.append(f'"{ch}"')

        kind = str(r["token_kind"])
        txt = str(r["token_text"])

        if kind == "chord":
            # stored without quotes
            parts.append(f'"{txt}"')
        else:
            parts.append(txt)

    return "".join(parts).strip()


def build_full_abc(
    name: str,
    tune_type: str,
    meter: str,
    mode_str: str,
    abc_body: str,
    x: int = 1,
    default_unit: str = "1/8",
) -> str:
    """Create a copy/paste-able ABC with minimal headers."""
    kline = mode_to_abc_key(mode_str)

    header = [
        f"X:{x}",
        f"T:{name}",
        f"R:{tune_type}",
        f"M:{meter}",
        f"L:{default_unit}",
        f"K:{kline}",
    ]
    return "\n".join(header) + "\n" + abc_body.strip() + "\n"


def _slot_event_index_map(
    notes_df: pd.DataFrame,
    slots_per_measure_by_key: Mapping[Tuple[int, int, int], int],
) -> Dict[Tuple[int, int, int, int], int]:
    """Map (setting, part, measure, slot_pos) -> first event_index for that slot.

    Slot advancement matches `features.beat_slots.build_beat_slots`: only note/rest
    tokens advance the slot position.
    """
    mapping: Dict[Tuple[int, int, int, int], int] = {}

    grp_cols = ["setting_id", "part", "measure_number"]
    for (sid, part, meas), g in notes_df.groupby(grp_cols, sort=False):
        sid_i, part_i, meas_i = int(sid), int(part), int(meas)
        g = g.sort_values("event_index")

        spm = slots_per_measure_by_key.get((sid_i, part_i, meas_i))
        if spm is None:
            # fallback to meter numerator
            try:
                meter = str(g.iloc[0].get("meter") or "4/4")
                spm = meter_to_slots(meter)
            except Exception:
                spm = 4

        slot_pos = 1
        for _, r in g.iterrows():
            kind = str(r["token_kind"])
            ev = int(r["event_index"])

            if kind in {"note", "rest"}:
                key = (sid_i, part_i, meas_i, int(slot_pos))
                mapping.setdefault(key, ev)
                slot_pos += 1
                if slot_pos > int(spm):
                    slot_pos = 1

    return mapping


def build_chord_inserts_from_predictions(
    beat_pred: pd.DataFrame,
    notes_df: pd.DataFrame,
    chord_label_col: str = "pred_chord_nashville",
    present_col: str = "pred_chord_present",
) -> Dict[int, List[str]]:
    """Map predicted chords to event_index positions for ABC injection.

    Expects `beat_pred` to contain: setting_id, part, measure_number, slot_position,
    slots_per_measure, mode.
    """
    by_event: Dict[int, List[str]] = defaultdict(list)
    if len(beat_pred) == 0:
        return by_event

    # slots_per_measure per measure
    spm_by_measure = (
        beat_pred.groupby(["setting_id", "part", "measure_number"], sort=False)["slots_per_measure"].first().to_dict()
    )
    slot_to_event = _slot_event_index_map(notes_df, spm_by_measure)

    for _, r in beat_pred.iterrows():
        if int(r.get(present_col, 0) or 0) != 1:
            continue

        mode_str = str(r.get("mode", ""))
        nash = str(r.get(chord_label_col, ""))
        if not mode_str or not nash:
            continue

        tonic_pc, km = parse_mode(mode_str)
        ch = nashville_to_chord_symbol(tonic_pc, km, nash)
        if not ch:
            continue

        sid = int(r["setting_id"])
        part = int(r["part"])
        meas = int(r["measure_number"])
        slot_pos = int(r["slot_position"])

        ev = slot_to_event.get((sid, part, meas, slot_pos))
        if ev is None:
            continue
        by_event[int(ev)].append(ch)

    return by_event
