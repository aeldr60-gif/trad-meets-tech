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

from trad_chords.models.training_data import make_training_frames
from trad_chords.models.baseline import train_baseline, BaselineModels

import json
from pathlib import Path


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

@app.command("train")
def train_cmd(config: str = DEFAULT_CONFIG):
    cfg = load_config(config)

    beat = pd.read_csv(cfg.artifacts.beat_slots_csv)
    chordy = pd.read_csv(cfg.artifacts.chordy_index_csv)
    chordy_ids = set(chordy["tune_id"].tolist())
    beat = beat[beat["tune_id"].isin(chordy_ids)].copy()

    X, y_place, X_tone, y_tone = make_training_frames(beat)

    # 🔒 SAFETY CHECK
    if y_place.nunique() < 2:
        raise ValueError(
            f"Placement labels have only one class: {y_place.unique().tolist()}. "
            "This means chord tokens were not detected in beat slots."
        )

    models = train_baseline(X, y_place, X_tone, y_tone, seed=42)
    models.save(cfg.paths.model_dir)

    print(f"Trained on {len(beat):,} beat slots")



@app.command("evaluate-selfcheck")
def evaluate_selfcheck_cmd(config: str = DEFAULT_CONFIG):
    import json
    from pathlib import Path
    import pandas as pd

    cfg = load_config(config)

    beat = pd.read_csv(cfg.artifacts.beat_slots_csv)
    chordy = pd.read_csv(cfg.artifacts.chordy_index_csv)
    chordy_ids = set(chordy["tune_id"].tolist())
    beat = beat[beat["tune_id"].isin(chordy_ids)].copy()

    X, y_place, X_tone, y_tone = make_training_frames(beat)
    models = BaselineModels.load(cfg.paths.model_dir)

    y_place_pred = models.placement.predict(X)
    placement_acc = float((y_place_pred == y_place).mean())

    tone_mask = (y_place == 1)
    if int(tone_mask.sum()) > 0:
        y_tone_pred = models.tone.predict(X.loc[tone_mask])
        tone_acc = float((y_tone_pred == y_tone.values).mean())
    else:
        tone_acc = 0.0

    print(f"Self-check placement accuracy: {placement_acc:.3f}")
    print(f"Self-check tone accuracy (on true chord slots): {tone_acc:.3f}")

    # Write metrics
    (cfg.paths.outputs_dir / "evaluation").mkdir(parents=True, exist_ok=True)

    metrics = {
        "placement_accuracy": placement_acc,
        "tone_accuracy": tone_acc,
        "n_rows": int(len(beat)),
        "n_chordy_tunes": int(len(chordy_ids)),
    }

    metrics_json = cfg.paths.outputs_dir / "evaluation" / "selfcheck_metrics.json"
    summary_csv = cfg.paths.outputs_dir / "evaluation" / "selfcheck_summary.csv"

    metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame([metrics]).to_csv(summary_csv, index=False)

    print(f"Wrote {metrics_json}")
    print(f"Wrote {summary_csv}")
