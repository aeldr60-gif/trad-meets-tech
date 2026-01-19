import pandas as pd

import typer
from rich import print

from trad_chords.config import load_config
from trad_chords.io.fetch import fetch_thesession_csvs

from trad_chords.io.loaders import load_tunes_csv, load_popularity_csv
from trad_chords.features.index_builder import build_jigs_reels_index, write_index
from trad_chords.features.note_table import build_notes_table, write_notes_table
from trad_chords.features.beat_slots import build_beat_slots, write_beat_slots
from trad_chords.utils.cache import should_skip

from trad_chords.features.splitter import split_chordy_chordless, write_df
from trad_chords.utils.cache import should_skip





app = typer.Typer(add_completion=False)
DEFAULT_CONFIG = "configs/default.yaml"


@app.command()
def hello():
    print("trad-chords CLI is working ✅")


@app.command("fetch-data")
def fetch_data(config: str = DEFAULT_CONFIG):
    cfg = load_config(config)
    fetch_thesession_csvs(
        tunes_url=cfg.sources.tunes_url,
        popularity_url=cfg.sources.popularity_url,
        tunes_dest=cfg.paths.raw_tunes_csv,
        popularity_dest=cfg.paths.raw_popularity_csv,
    )
    print("Downloaded:")
    print(f"- {cfg.paths.raw_tunes_csv}")
    print(f"- {cfg.paths.raw_popularity_csv}")

@app.command("load-data")
def load_data(config: str = DEFAULT_CONFIG):
    cfg = load_config(config)
    tunes = load_tunes_csv(cfg.paths.raw_tunes_csv)
    pop = load_popularity_csv(cfg.paths.raw_popularity_csv)

    print(f"Tunes rows: {len(tunes):,}")
    print(f"Tunes columns: {list(tunes.columns)}")
    print(f"Popularity rows: {len(pop):,}")
    print(f"Popularity columns: {list(pop.columns)}")
    print("Sample tune types:", sorted(tunes["type"].dropna().astype(str).str.lower().unique())[:20])

@app.command("build-index")
def build_index(config: str = DEFAULT_CONFIG):
    cfg = load_config(config)

    if should_skip(cfg.artifacts.jigs_reels_index_csv, cfg.run.overwrite):
        print(f"Skipping build-index (already exists): {cfg.artifacts.jigs_reels_index_csv}")
        print("Tip: set run.overwrite: true in configs/default.yaml to regenerate.")
        return

    tunes = load_tunes_csv(cfg.paths.raw_tunes_csv)
    pop = load_popularity_csv(cfg.paths.raw_popularity_csv)

    df = build_jigs_reels_index(
        tunes=tunes,
        popularity=pop,
        tune_types=cfg.pipeline.tune_types,
        min_tunebooks=cfg.pipeline.min_tunebooks,
        top_n=cfg.pipeline.top_n,
    )

    write_index(df, cfg.artifacts.jigs_reels_index_csv)
    print(f"Wrote {len(df):,} rows -> {cfg.artifacts.jigs_reels_index_csv}")
    print(f"top_n={cfg.pipeline.top_n} | min_tunebooks={cfg.pipeline.min_tunebooks}")
    print(df.head(5))


@app.command("build-notes-table")
def build_notes_table_cmd(config: str = DEFAULT_CONFIG):
    cfg = load_config(config)
    index_df = pd.read_csv(cfg.artifacts.jigs_reels_index_csv)

    if should_skip(cfg.artifacts.notes_table_csv, cfg.run.overwrite):
        print(f"Skipping build-notes-table (already exists): {cfg.artifacts.notes_table_csv}")
        return

    notes = build_notes_table(index_df)
    write_notes_table(notes, cfg.artifacts.notes_table_csv)

    print(f"Wrote notes table: {len(notes):,} rows -> {cfg.artifacts.notes_table_csv}")
    print(notes.head(10))

@app.command("build-beat-slots")
def build_beat_slots_cmd(config: str = DEFAULT_CONFIG):
    cfg = load_config(config)

    if should_skip(cfg.artifacts.beat_slots_csv, cfg.run.overwrite):
        print(f"Skipping build-beat-slots (already exists): {cfg.artifacts.beat_slots_csv}")
        return

    # notes_table is huge; only load needed columns
    usecols = [
        "tune_id","name","type","meter","mode","tunebooks",
        "part","measure_number","event_index","token_kind","token_text"
    ]
    notes = pd.read_csv(cfg.artifacts.notes_table_csv, usecols=usecols)

    slots = build_beat_slots(notes)
    write_beat_slots(slots, cfg.artifacts.beat_slots_csv)

    print(f"Wrote beat slots: {len(slots):,} rows -> {cfg.artifacts.beat_slots_csv}")
    print(slots.head(10))

@app.command("split-index")
def split_index_cmd(config: str = DEFAULT_CONFIG):
    cfg = load_config(config)

    if should_skip(cfg.artifacts.chordy_index_csv, cfg.run.overwrite) and should_skip(cfg.artifacts.chordless_index_csv, cfg.run.overwrite):
        print("Skipping split-index (already exists).")
        return

    index_df = pd.read_csv(cfg.artifacts.jigs_reels_index_csv)

    chordy, chordless = split_chordy_chordless(index_df)

    write_df(chordy, cfg.artifacts.chordy_index_csv)
    write_df(chordless, cfg.artifacts.chordless_index_csv)

    print(f"Chordy tunes: {len(chordy):,} -> {cfg.artifacts.chordy_index_csv}")
    print(f"Chordless tunes: {len(chordless):,} -> {cfg.artifacts.chordless_index_csv}")
