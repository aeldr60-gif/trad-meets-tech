from __future__ import annotations

from pathlib import Path
import pandas as pd
from typing import Iterable, Optional

import re

CHORD_RE = re.compile(r'"[A-G][#b]?(?:m|min|maj|dim|aug|7)?[^"]*"')



def build_jigs_reels_index(
    tunes: pd.DataFrame,
    popularity: pd.DataFrame,
    tune_types: Iterable[str],
    min_tunebooks: int = 0,
    top_n: Optional[int] = None,
) -> pd.DataFrame:
    """
    Build an index of jigs/reels joined with popularity.

    This function ENFORCES top_n at the very end to guarantee correctness.
    """

    # Normalize tune types
    tune_types = {t.lower() for t in tune_types}

    # Filter to requested tune types
    df = tunes.copy()
    df["type"] = df["type"].astype(str).str.lower()
    df = df[df["type"].isin(tune_types)]
    
    # Which tunes have chords?
    df["has_chords"] = df["abc"].fillna("").str.contains(CHORD_RE)

    # Join popularity
    pop = popularity[["tune_id", "tunebooks"]]
    df = df.merge(pop, on="tune_id", how="left")

    # Fill missing popularity
    df["tunebooks"] = df["tunebooks"].fillna(0).astype(int)

    # Apply minimum popularity filter
    if min_tunebooks > 0:
        df = df[df["tunebooks"] >= min_tunebooks]

    # Sort by popularity DESC
    df = df.sort_values("tunebooks", ascending=False)

    # 🔑 ENFORCE top_n HERE (after all filtering + sorting)
    if top_n is not None:
        df = df.head(int(top_n))

    # Reset index for cleanliness
    df = df.reset_index(drop=True)

    return df


def write_index(df: pd.DataFrame, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
