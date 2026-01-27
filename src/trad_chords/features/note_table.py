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

    # IMPORTANT: multiple settings exist for the same tune_id on TheSession.
    # Throughout the pipeline we therefore treat *setting_id* as the primary identifier.
    for _, r in index_df.iterrows():
        tune_id = r.get("tune_id")
        setting_id = r.get("setting_id")
        name = r.get("name")
        ttype = r.get("type")
        meter = r.get("meter")
        mode = r.get("mode")
        tunebooks = r.get("tunebooks", 0)
        abc_raw = str(r.get("abc", "") or "")
        abc = remove_decorations(abc_raw)

        active_chord: Optional[str] = None


        measure = 1
        event_i = 0
        seen_any_notes_in_measure = False

        part = 1
        measure_number = 1

        measures_in_part = 0          # count completed measures inside the current part
        started_music = False         # becomes True after we see any note/rest/chord (musical content)

        PART_START_BARS = {"|:", "||:", "[|", "::"}   # bars that often indicate a new “part” start
        MEASURE_END_BARS = {"|", "||", ":|", "|]", "[|", "::", "|:", "||:"}  # anything you treat as a measure boundary


        for tok in tokenize_abc(abc):
            event_i += 1
            # 1) Endings should NOT create new parts
            if tok.kind == "ending":
                # keep measure/part unchanged
                continue

            # Track that we’ve actually begun musical content (so we don’t create “part 2, measure 1”)
            if tok.kind in {"note", "rest"}:
                started_music = True
            elif tok.kind == "chord":
                # chord markers count as "content" too; they shouldn't trigger part bumps at the very start
                started_music = True

            if tok.kind == "bar":
                bt = tok.text

                # 2) Count measures when you hit a measure boundary.
                #    If your code increments measure_number elsewhere, adapt accordingly.
                if bt in MEASURE_END_BARS:
                    measures_in_part += 1
                    measure_number += 1

                # 3) Only start a new part when:
                #    - we've already started the tune (saw notes/rests/chords)
                #    - we have a “real” part length already (>= 2 measures)
                #    - and we hit a part-start bar token
                #
                # This prevents:
                #    - leading "||:" or "|:" from causing part 2 at measure 1
                #    - tiny 1–2-measure “parts” created by endings/repeats noise
                if (bt in PART_START_BARS) and started_music and (measures_in_part >= 2):
                    part += 1
                    measures_in_part = 0  # reset for the new part
                    # NOTE: do NOT reset measure_number; measure_number stays global in the tune

                # continue to your existing handling (e.g., active chord reset) if any
                continue
            if tok.kind == "bar":
                # A "part" in session tune settings typically begins at:
                # - start of tune
                # - repeat starts/ends or double bars
                if tok.text in {"||", "|:", ":|", ":||", "||:"}:
                    part += 1

                # Only advance the measure counter if we've actually seen notes since the last bar.
                # (This avoids creating a phantom first measure when the abc begins with a leading bar token like "|:".)
                if seen_any_notes_in_measure:
                    measure += 1
                    seen_any_notes_in_measure = False

            if tok.kind == "note":
                seen_any_notes_in_measure = True

            if tok.kind == "chord":
                active_chord = tok.text

            rows.append(
                {
                    "tune_id": tune_id,
                    "setting_id": setting_id,
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
