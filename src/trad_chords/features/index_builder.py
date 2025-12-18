from __future__ import annotations

from pathlib import Path
import pandas as pd


def build_jigs_reels_index(
    tunes: pd.DataFrame,
    popularity: pd.DataFrame,
    tune_types: list[str],
    min_tunebooks: int = 0,
    top_n: int | None = None,
) -> pd.DataFrame:
    """
    Build an index of tunes filtered to target tune types, joined with popularity.

    Output columns (minimum):
      tune_id, name, type, meter, mode, tunebooks, has_chords, abc
    """
    t = tunes.copy()

    # Normalize types
    t["type_norm"] = t["type"].astype(str).str.strip().str.lower()

    # Filter to desired types
    keep = set([x.lower() for x in tune_types])
    t = t[t["type_norm"].isin(keep)].copy()

    # Popularity join (ensure tune_id is numeric where possible)
    pop = popularity.copy()
    pop["tune_id"] = pd.to_numeric(pop["tune_id"], errors="coerce")
    t["tune_id"] = pd.to_numeric(t["tune_id"], errors="coerce")

    t = t.merge(pop[["tune_id", "tunebooks"]], on="tune_id", how="left")
    t["tunebooks"] = t["tunebooks"].fillna(0).astype(int)

    # Detect chord presence in ABC: common syntax is "A" in quotes, e.g. ""Am"" or ""G""
    # We'll treat any quoted chord token as chords present.
    abc = t["abc"].fillna("").astype(str)
    t["has_chords"] = abc.str.contains(r'""[^"]+""', regex=True)

    # Apply popularity threshold
    t = t[t["tunebooks"] >= int(min_tunebooks)].copy()

    # Optional: keep only top N by tunebooks
    if top_n is not None:
        t = t.sort_values(["tunebooks", "tune_id"], ascending=[False, True]).head(int(top_n)).copy()
    else:
        t = t.sort_values(["tunebooks", "tune_id"], ascending=[False, True]).copy()

    # Select/standardize columns
    cols = []
    for c in ["tune_id", "name", "type", "meter", "mode", "tunebooks", "has_chords", "abc"]:
        if c in t.columns:
            cols.append(c)
    return t[cols].reset_index(drop=True)


def write_index(df: pd.DataFrame, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
