from __future__ import annotations

from pathlib import Path
import pandas as pd

from trad_chords.music.theory import (
    abc_note_to_pitch_class,
    parse_mode_string,
    split_key_and_mode,
    pitch_class_to_degree_bin,
    parse_chord_symbol,
    chord_to_nashville,
)



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
      - **Key-agnostic** scale-degree histograms (deg_1..deg_7 plus in-between bins deg_1_5..deg_7_5)
      - whether a chord token occurs at this slot (supervised placement)
      - chord label (raw) + a simple Nashville-style chord root degree (key-agnostic)

    Notes are interpreted in context of the tune's key/mode so that (for example) C in D major
    is treated as the 7th degree (C#) vs a non-diatonic tone.
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
    # IMPORTANT: setting_id is the unique identifier for a specific setting of a tune.
    group_cols = ["setting_id", "part", "measure_number"]
    base_cols = [
        "tune_id",
        "setting_id",
        "name",
        "type",
        "meter",
        "mode",
        "tunebooks",
        "part",
        "measure_number",
    ]

    for (setting_id, part, measure), g in df.groupby(group_cols, sort=False):
        g = g.sort_values("event_index")
        slots_per_measure = int(g["slots_per_measure"].iloc[0]) if "slots_per_measure" in g else 4

        # Key + mode handling
        mode_str = str(g["mode"].iloc[0]) if "mode" in g.columns else ""
        key_mode = parse_mode_string(mode_str)

        # These are for your convenience columns
        tonic, music_mode = split_key_and_mode(mode_str)


        slot_pos = 1
        # slot accumulators per slot
        slot_data = {}

        DEG_COLS = [
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

        def ensure_slot(sp: int):
            if sp not in slot_data:
                slot_data[sp] = {
                    "slot_position": sp,
                    "slots_per_measure": slots_per_measure,
                    **{c: 0 for c in DEG_COLS},
                    "rests": 0,
                    "has_chord_here": 0,
                    "chord_label": None,
                    "chord_root_degree": None,
                    "chord_root_pc": None,
                    "chord_quality": None,
                    "chord_nashville": None,
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

                # Map chord root into a Nashville-style degree (key-agnostic)
                root_pc, quality = parse_chord_symbol(txt)
                slot_data[slot_pos]["chord_root_pc"] = root_pc
                slot_data[slot_pos]["chord_quality"] = quality

                if root_pc is not None and key_mode is not None:
                    root_deg = pitch_class_to_degree_bin(root_pc, key_mode)   # e.g., "deg_6" or "deg_6_5"
                    slot_data[slot_pos]["chord_root_degree"] = root_deg

                    deg_short, q = chord_to_nashville(txt, key_mode)          # e.g., ("6", "min") or ("2_5", "maj")
                    if deg_short and q:
                        slot_data[slot_pos]["chord_nashville"] = f"{deg_short}:{q}"


            elif kind == "rest":
                slot_data[slot_pos]["rests"] += 1
                slot_pos += int(row.get("dur_slots", 1))
            elif kind == "note":

                # Convert ABC note to pitch class, then to key-agnostic degree bin.
                pc = abc_note_to_pitch_class(txt)
                if pc is not None and key_mode is not None:
                    deg_col = pitch_class_to_degree_bin(pc, key_mode)  # returns "deg_#" or "deg_#_5"
                    if deg_col in slot_data[slot_pos]:
                        slot_data[slot_pos][deg_col] += 1

                slot_pos += int(row.get("dur_slots", 1))

            # clamp to measure slot range (prevents drift when durations are messy)
            if slot_pos > slots_per_measure:
                slot_pos = slots_per_measure

        # emit slots 1..slots_per_measure (even if empty)
        meta = {c: g[c].iloc[0] for c in base_cols if c in g.columns}
        # Add explicit key + music_mode columns for convenience
        meta["key"] = tonic
        meta["music_mode"] = music_mode
        for sp in range(1, slots_per_measure + 1):
            ensure_slot(sp)
            out = dict(meta)
            out.update(slot_data[sp])
            out_rows.append(out)

    return pd.DataFrame(out_rows)


def write_beat_slots(df: pd.DataFrame, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
