from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from trad_chords.abc.cleaning import remove_decorations
from trad_chords.abc.parser import tokenize_abc, AbcToken


def build_notes_table(index_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert jigs/reels index rows into a flat per-event table.

    Initial schema (we'll expand later):
      tune_id, name, type, meter, mode, tunebooks
      part (placeholder), measure_number, event_index
      token_kind, token_text
      active_chord (last seen chord token)
    """
    rows: List[Dict[str, Any]] = []

    for _, r in index_df.iterrows():
        tune_id = r.get("tune_id")
        name = r.get("name")
        ttype = r.get("type")
        meter = r.get("meter")
        mode = r.get("mode")
        tunebooks = r.get("tunebooks", 0)
        abc_raw = str(r.get("abc", "") or "")
        abc = remove_decorations(abc_raw)

        active_chord: Optional[str] = None
        measure = 1
        part = 1
        event_i = 0

        for tok in tokenize_abc(abc):
            event_i += 1

            if tok.kind == "bar":
                # crude part detection: treat "||" as a part boundary sometimes
                if tok.text == "||":
                    part += 1
                measure += 1

            if tok.kind == "chord":
                active_chord = tok.text

            rows.append(
                {
                    "tune_id": tune_id,
                    "name": name,
                    "type": ttype,
                    "meter": meter,
                    "mode": mode,
                    "tunebooks": tunebooks,
                    "part": part,
                    "measure_number": measure,
                    "event_index": event_i,
                    "token_kind": tok.kind,
                    "token_text": tok.text,
                    "active_chord": active_chord,
                }
            )

    return pd.DataFrame(rows)


def write_notes_table(df: pd.DataFrame, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
