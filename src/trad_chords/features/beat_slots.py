from __future__ import annotations

from pathlib import Path
from typing import Tuple
import pandas as pd


def _meter_to_beats(meter: str) -> int:
    """
    Convert meter to number of beats per measure for our slotting.
    - reels: 4/4 -> 4 beats
    - jigs: 6/8 -> 6 "eighth-note slots" (we’ll treat as 6 slots)
    """
    m = str(meter).strip()
    if "/" not in m:
        return 4
    num, den = m.split("/", 1)
    try:
        num_i = int(num)
        den_i = int(den)
    except Exception:
        return 4
    # For 6/8 we want 6 slots; for 9/8 -> 9; for 12/8 -> 12
    return num_i


def _note_length_to_slots(token_text: str) -> int:
    """
    Very pragmatic duration parsing:
      E2 -> 2 slots
      E  -> 1 slot
      E/ -> 0.5 slot -> we round up to 1 (keeps alignment stable)
      E/2 -> 0.5 -> round up to 1
    """
    s = str(token_text)
    # strip leading accidentals and note letter/octaves
    # keep the trailing length bits (digits or /)
    length_part = ""
    for ch in reversed(s):
        if ch.isdigit() or ch == "/":
            length_part = ch + length_part
        else:
            break

    if length_part == "":
        return 1
    if length_part.isdigit():
        return max(1, int(length_part))
    if length_part == "/":
        return 1
    if length_part.startswith("/"):
        return 1
    return 1


def build_beat_slots(notes_table: pd.DataFrame) -> pd.DataFrame:
    """
    Build one row per beat-slot within each measure.

    Each slot aggregates:
      - pitch-class histogram-ish signals (counts of A..G)
      - whether a chord token occurs at this slot (supervised placement)
      - chord label if present (supervised chord class)

    This intentionally trades musical precision for ML-ready consistency.
    """
    df = notes_table.copy()

    # Only keep tokens relevant to timing
    df = df[df["token_kind"].isin(["note", "rest", "bar", "chord"])].copy()

    # Normalize meter -> slots per measure
    df["slots_per_measure"] = df["meter"].map(_meter_to_beats).astype(int)

    # Establish a running slot position within the measure
    # Reset on bar tokens.
    df["dur_slots"] = 0
    note_mask = df["token_kind"].isin(["note", "rest"])
    df.loc[note_mask, "dur_slots"] = df.loc[note_mask, "token_text"].map(_note_length_to_slots).astype(int)

    # We’ll compute slot_position by scanning within each tune+part+measure
    out_rows = []
    group_cols = ["tune_id", "part", "measure_number"]
    base_cols = ["tune_id", "name", "type", "meter", "mode", "tunebooks", "part", "measure_number"]

    for (tune_id, part, measure), g in df.groupby(group_cols, sort=False):
        g = g.sort_values("event_index")
        slots_per_measure = int(g["slots_per_measure"].iloc[0]) if "slots_per_measure" in g else 4

        slot_pos = 1
        # slot accumulators per slot
        slot_data = {}

        def ensure_slot(sp: int):
            if sp not in slot_data:
                slot_data[sp] = {
                    "slot_position": sp,
                    "slots_per_measure": slots_per_measure,
                    "notes_A": 0, "notes_B": 0, "notes_C": 0, "notes_D": 0, "notes_E": 0, "notes_F": 0, "notes_G": 0,
                    "rests": 0,
                    "has_chord_here": 0,
                    "chord_label": None,
                }

        for _, row in g.iterrows():
            kind = row["token_kind"]
            txt = str(row["token_text"])

            if kind == "bar":
                # bar token: treat as measure boundary already handled by group
                continue

            ensure_slot(slot_pos)

            if kind == "chord":
                # placement label: chord occurs "at" current slot
                slot_data[slot_pos]["has_chord_here"] = 1
                slot_data[slot_pos]["chord_label"] = txt

            elif kind == "rest":
                slot_data[slot_pos]["rests"] += 1
                slot_pos += int(row.get("dur_slots", 1))
            elif kind == "note":
                # crude pitch class: just A..G letter ignoring octave/accidental for now
                letter = None
                for ch in txt:
                    if ch.upper() in ["A","B","C","D","E","F","G"]:
                        letter = ch.upper()
                        break
                if letter:
                    slot_data[slot_pos][f"notes_{letter}"] += 1
                slot_pos += int(row.get("dur_slots", 1))

            # clamp to measure slot range (prevents drift when durations are messy)
            if slot_pos > slots_per_measure:
                slot_pos = slots_per_measure

        # emit slots 1..slots_per_measure (even if empty)
        meta = {c: g[c].iloc[0] for c in base_cols if c in g.columns}
        for sp in range(1, slots_per_measure + 1):
            ensure_slot(sp)
            out = dict(meta)
            out.update(slot_data[sp])
            out_rows.append(out)

    return pd.DataFrame(out_rows)


def write_beat_slots(df: pd.DataFrame, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
