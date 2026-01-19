from __future__ import annotations
from pathlib import Path
import pandas as pd


def split_chordy_chordless(index_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the jigs/reels index into chordy vs chordless using boolean has_chords.
    """
    if "has_chords" not in index_df.columns:
        raise ValueError("index_df is missing required column: has_chords")

    chordy = index_df[index_df["has_chords"] == True].copy()
    chordless = index_df[index_df["has_chords"] == False].copy()

    return chordy.reset_index(drop=True), chordless.reset_index(drop=True)


def write_df(df: pd.DataFrame, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
