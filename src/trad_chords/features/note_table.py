from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd

from trad_chords.abc.parser import tokenize_abc


# Bar tokens that we treat as a *part* boundary (new section).
PART_BOUNDARY_BARS = {"||", "||:", ":||", "::", "[|", "|]"}


def build_notes_table(index_df: pd.DataFrame) -> pd.DataFrame:
    """Expand ABC bodies into a token-level notes table.

    Each row in the returned table is one token (note/rest/chord/bar/etc.) from
    the ABC body. We also attach contextual metadata (tune_id/setting_id/etc.)
    from the index.

    Important invariants:
    - We treat *setting_id* as the primary ID (multiple settings can share the
      same tune_id).
    - Parts start at 1. A tune can never begin at part 2.
    - 1st/2nd/3rd endings tokens (e.g., [1, [2) are *not* treated as parts.
    """

    required = {"tune_id", "setting_id", "name", "type", "meter", "mode", "abc", "tunebooks"}
    missing = [c for c in required if c not in index_df.columns]
    if missing:
        raise ValueError(f"index_df missing required columns: {missing}")

    out_rows: List[dict] = []

    for _, row in index_df.iterrows():
        abc_body = str(row.get("abc") or "")

        base = {
            "tune_id": int(row["tune_id"]),
            "setting_id": int(row["setting_id"]),
            "name": row.get("name"),
            "type": row.get("type"),
            "meter": row.get("meter"),
            "mode": row.get("mode"),
            "tunebooks": int(row.get("tunebooks") or 0),
        }

        part = 1
        measure = 1
        event_index = 0
        started = False  # becomes True once we see first note/rest/chord token
        active_chord = None

        for tok in tokenize_abc(abc_body):
            # Track whether we've actually started musical content
            if tok.kind in {"note", "rest", "chord"}:
                started = True

            # Bar tokens advance measures, and sometimes parts.
            if tok.kind == "bar":
                # If the tune *begins* with a repeat/double-bar marker (e.g., ||:),
                # do not advance part/measure.
                if not started and measure == 1 and event_index == 0:
                    pass
                else:
                    if tok.text in PART_BOUNDARY_BARS:
                        part += 1
                        measure = 1
                    else:
                        measure += 1

            # Chord tokens update the active chord
            if tok.kind == "chord":
                active_chord = tok.text

            out_rows.append(
                {
                    **base,
                    "part": part,
                    "measure_number": measure,
                    "event_index": event_index,
                    "token_kind": tok.kind,
                    "token_text": tok.text,
                    "active_chord": active_chord,
                }
            )

            event_index += 1

    return pd.DataFrame(out_rows)


def write_notes_table(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)
