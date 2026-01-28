from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import typer
from rich import print

from trad_chords.config import load_config
from trad_chords.io.fetch import fetch_thesession_csvs
from trad_chords.io.loaders import load_tunes_csv, load_popularity_csv
from trad_chords.features.index_builder import build_jigs_reels_index, write_index
from trad_chords.features.note_table import build_notes_table, write_notes_table
from trad_chords.features.beat_slots import build_beat_slots, write_beat_slots
from trad_chords.features.splitter import split_chordy_chordless, write_df
from trad_chords.utils.cache import should_skip
from trad_chords.models.training_data import make_training_frames
from trad_chords.models.baseline import train_baseline, BaselineModels
from trad_chords.inference.harmonize import harmonize_chordless
from trad_chords.models.feature_sets import FEATURE_SETS, get_feature_cols

"""
Command line interface for the entire trad chords processing and modeling pipeline.

This Typer based CLI ties together all major stages of the project: fetching raw
TheSession CSVs, inspecting loaded data, building tune indexes, expanding ABC
into token level notes tables, generating beat slot features, splitting chordy
vs. chordless settings, training the baseline chord prediction models, evaluating
their self consistency, sweeping feature set variants, and finally harmonizing
chordless tunes into copy pasteable ABC.

Each command wraps a well defined pipeline step, reads paths and parameters from
the project config, and writes reproducible artifacts under the configured output
directories. This script is the main entry point for running the full workflow
from raw data → features → models → inferred harmonizations.
"""




app = typer.Typer(add_completion=False)
DEFAULT_CONFIG = "configs/default.yaml"

def _preview_setting(df: pd.DataFrame, setting_id: int | None, n: int, label: str) -> None:
    if setting_id is None:
        return
    if "setting_id" not in df.columns:
        print(f"[yellow]Preview skipped ({label}): no setting_id column[/yellow]")
        return
    sub = df[df["setting_id"] == setting_id]
    if sub.empty:
        print(f"[yellow]Preview ({label}): no rows for setting_id={setting_id}[/yellow]")
        return
    print(f"\n[bold]Preview ({label}) setting_id={setting_id}[/bold]")
    print(sub.head(n))


def _preview_features_for_setting(X: pd.DataFrame, df: pd.DataFrame, setting_id: int | None, n: int, label: str) -> None:
    """Print feature rows corresponding to a setting_id (X and df share index)."""
    if setting_id is None:
        return
    if "setting_id" not in df.columns:
        return
    idx = df.index[df["setting_id"] == setting_id]
    if len(idx) == 0:
        print(f"[yellow]Preview ({label} features): no rows for setting_id={setting_id}[/yellow]")
        return
    print(f"\n[bold]Preview ({label} features) setting_id={setting_id}[/bold]")
    print(X.loc[idx].head(n))



@app.command()
def hello() -> None:
    print("trad-chords CLI is working ✅")


@app.command("fetch-data")
def fetch_data(config: str = DEFAULT_CONFIG) -> None:
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

    preview_setting_id: int | None = typer.Option(None, "--preview-setting-id", help="Print head() for this setting_id."),
    preview_n: int = typer.Option(5, "--preview-n", help="Rows to print for previews."),



@app.command("load-data")
def load_data(config: str = DEFAULT_CONFIG) -> None:
    cfg = load_config(config)
    tunes = load_tunes_csv(cfg.paths.raw_tunes_csv)
    pop = load_popularity_csv(cfg.paths.raw_popularity_csv)

    print(f"Tunes rows: {len(tunes):,}")
    print(f"Tunes columns: {list(tunes.columns)}")
    print(f"Popularity rows: {len(pop):,}")
    print(f"Popularity columns: {list(pop.columns)}")
    if "type" in tunes.columns:
        print("Sample tune types:", sorted(tunes["type"].dropna().astype(str).str.lower().unique())[:20])

    preview_setting_id: int | None = typer.Option(None, "--preview-setting-id", help="Print head() for this setting_id."),
    preview_n: int = typer.Option(5, "--preview-n", help="Rows to print for previews."),


@app.command("build-index")
def build_index_cmd(config: str = DEFAULT_CONFIG) -> None:
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

    preview_setting_id: int | None = typer.Option(None, "--preview-setting-id", help="Print head() for this setting_id."),
    preview_n: int = typer.Option(5, "--preview-n", help="Rows to print for previews."),



@app.command("build-notes-table")
def build_notes_table_cmd(config: str = DEFAULT_CONFIG) -> None:
    cfg = load_config(config)
    index_df = pd.read_csv(cfg.artifacts.jigs_reels_index_csv)

    if should_skip(cfg.artifacts.notes_table_csv, cfg.run.overwrite):
        print(f"Skipping build-notes-table (already exists): {cfg.artifacts.notes_table_csv}")
        return

    notes = build_notes_table(index_df)
    write_notes_table(notes, cfg.artifacts.notes_table_csv)

    print(f"Wrote notes table: {len(notes):,} rows -> {cfg.artifacts.notes_table_csv}")
    print(notes.head(10))

    preview_setting_id: int | None = typer.Option(None, "--preview-setting-id", help="Print head() for this setting_id."),
    preview_n: int = typer.Option(5, "--preview-n", help="Rows to print for previews."),



@app.command("build-beat-slots")
def build_beat_slots_cmd(config: str = DEFAULT_CONFIG) -> None:
    cfg = load_config(config)

    if should_skip(cfg.artifacts.beat_slots_csv, cfg.run.overwrite):
        print(f"Skipping build-beat-slots (already exists): {cfg.artifacts.beat_slots_csv}")
        return

    # notes_table can be large; only load needed columns
    usecols = [
        "tune_id",
        "setting_id",
        "name",
        "type",
        "meter",
        "mode",
        "tunebooks",
        "part",
        "measure_number",
        "event_index",
        "token_kind",
        "token_text",
        "active_chord",
    ]
    notes = pd.read_csv(cfg.artifacts.notes_table_csv, usecols=usecols)

    slots = build_beat_slots(notes)
    write_beat_slots(slots, cfg.artifacts.beat_slots_csv)

    print(f"Wrote beat slots: {len(slots):,} rows -> {cfg.artifacts.beat_slots_csv}")
    print(slots.head(10))

    preview_setting_id: int | None = typer.Option(None, "--preview-setting-id", help="Print head() for this setting_id."),
    preview_n: int = typer.Option(5, "--preview-n", help="Rows to print for previews."),



@app.command("split-index")
def split_index_cmd(config: str = DEFAULT_CONFIG) -> None:
    cfg = load_config(config)

    if should_skip(cfg.artifacts.chordy_index_csv, cfg.run.overwrite) and should_skip(
        cfg.artifacts.chordless_index_csv, cfg.run.overwrite
    ):
        print("Skipping split-index (already exists).")
        return

    index_df = pd.read_csv(cfg.artifacts.jigs_reels_index_csv)
    chordy, chordless = split_chordy_chordless(index_df)

    write_df(chordy, cfg.artifacts.chordy_index_csv)
    write_df(chordless, cfg.artifacts.chordless_index_csv)

    print(f"Chordy settings: {len(chordy):,} -> {cfg.artifacts.chordy_index_csv}")
    print(f"Chordless settings: {len(chordless):,} -> {cfg.artifacts.chordless_index_csv}")

    preview_setting_id: int | None = typer.Option(None, "--preview-setting-id", help="Print head() for this setting_id."),
    preview_n: int = typer.Option(5, "--preview-n", help="Rows to print for previews."),



@app.command("train")
def train_cmd(config: str = DEFAULT_CONFIG) -> None:
    cfg = load_config(config)

    beat = pd.read_csv(cfg.artifacts.beat_slots_csv)
    chordy = pd.read_csv(cfg.artifacts.chordy_index_csv)

    # Prefer setting_id (multiple settings per tune)
    if "setting_id" in chordy.columns and "setting_id" in beat.columns:
        chordy_ids = set(chordy["setting_id"].tolist())
        beat = beat[beat["setting_id"].isin(chordy_ids)].copy()
    else:
        chordy_ids = set(chordy["tune_id"].tolist())
        beat = beat[beat["tune_id"].isin(chordy_ids)].copy()

    X, y_place, X_tone, y_tone = make_training_frames(beat)

    # Safety check: placement must have at least 2 classes
    if y_place.nunique() < 2:
        raise ValueError(
            f"Placement labels have only one class: {y_place.unique().tolist()}. "
            "This means chord tokens were not detected in beat slots."
        )

    models = train_baseline(X, y_place, X_tone, y_tone, seed=42)
    models.save(cfg.paths.model_dir)

    print(f"Trained on {len(beat):,} beat slots")
    print(f"Wrote models -> {cfg.paths.model_dir}")

    preview_setting_id: int | None = typer.Option(None, "--preview-setting-id", help="Print head() for this setting_id."),
    preview_n: int = typer.Option(5, "--preview-n", help="Rows to print for previews."),



@app.command("evaluate-selfcheck")
def evaluate_selfcheck_cmd(config: str = DEFAULT_CONFIG) -> None:
    cfg = load_config(config)

    beat = pd.read_csv(cfg.artifacts.beat_slots_csv)
    chordy = pd.read_csv(cfg.artifacts.chordy_index_csv)

    if "setting_id" in chordy.columns and "setting_id" in beat.columns:
        chordy_ids = set(chordy["setting_id"].tolist())
        beat = beat[beat["setting_id"].isin(chordy_ids)].copy()
    else:
        chordy_ids = set(chordy["tune_id"].tolist())
        beat = beat[beat["tune_id"].isin(chordy_ids)].copy()

    X, y_place, X_tone, y_tone = make_training_frames(beat)
    models = BaselineModels.load(cfg.paths.model_dir)

    y_place_pred = models.placement.predict(X)
    placement_acc = float((y_place_pred == y_place).mean())

    tone_mask = y_place == 1
    if int(tone_mask.sum()) > 0:
        y_tone_pred = models.tone.predict(X.loc[tone_mask])
        tone_acc = float((y_tone_pred == y_tone.values).mean())
    else:
        tone_acc = 0.0

    print(f"Self-check placement accuracy: {placement_acc:.3f}")
    print(f"Self-check tone accuracy (on true chord slots): {tone_acc:.3f}")

    # Write metrics
    metrics = {
        "placement_accuracy": placement_acc,
        "tone_accuracy": tone_acc,
        "n_rows": int(len(beat)),
        "n_chordy_settings": int(len(chordy_ids)),
    }

    metrics_json = Path(cfg.artifacts.selfcheck_metrics_json)
    summary_csv = Path(cfg.artifacts.selfcheck_summary_csv)
    metrics_json.parent.mkdir(parents=True, exist_ok=True)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)

    metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame([metrics]).to_csv(summary_csv, index=False)

    print(f"Wrote {metrics_json}")
    print(f"Wrote {summary_csv}")

    preview_setting_id: int | None = typer.Option(None, "--preview-setting-id", help="Print head() for this setting_id."),
    preview_n: int = typer.Option(5, "--preview-n", help="Rows to print for previews."),



@app.command("sweep-feature-sets")
def sweep_feature_sets_cmd(
    config: str = DEFAULT_CONFIG,
    save_models: bool = typer.Option(
        False,
        "--save-models/--no-save-models",
        help="If set, write a copy of the trained models under outputs/models/feature_set_<name>/.",
    ),
    out_csv: str = typer.Option(
        "outputs/evaluation/feature_set_sweep.csv",
        help="Where to write the sweep results CSV.",
    ),
) -> None:
    """Train + self-check every FEATURE_SET and write a ranked summary CSV.

    This is meant for quick experimentation to see whether feature selection alone
    improves self-check placement/tone accuracy.
    """
    cfg = load_config(config)

    # Load once (large CSV). low_memory=False avoids mixed-type inference surprises.
    beat = pd.read_csv(cfg.artifacts.beat_slots_csv, low_memory=False)
    chordy = pd.read_csv(cfg.artifacts.chordy_index_csv)

    if "setting_id" in chordy.columns and "setting_id" in beat.columns:
        chordy_ids = set(chordy["setting_id"].tolist())
        beat = beat[beat["setting_id"].isin(chordy_ids)].copy()
    else:
        chordy_ids = set(chordy["tune_id"].tolist())
        beat = beat[beat["tune_id"].isin(chordy_ids)].copy()

    results = []

    for fs_name in FEATURE_SETS.keys():
        cols = get_feature_cols(fs_name)
        X, y_place, X_tone, y_tone = make_training_frames(beat, feature_set=fs_name)
        models = train_baseline(X, y_place, X_tone, y_tone, seed=42)

        y_place_pred = models.placement.predict(X)
        placement_acc = float((y_place_pred == y_place).mean())

        tone_mask = y_place == 1
        if int(tone_mask.sum()) > 0:
            y_tone_pred = models.tone.predict(X.loc[tone_mask])
            tone_acc = float((y_tone_pred == y_tone.values).mean())
        else:
            tone_acc = 0.0

        results.append(
            {
                "feature_set": fs_name,
                "n_features": int(len(cols)),
                "placement_accuracy": placement_acc,
                "tone_accuracy": tone_acc,
                "n_rows": int(len(beat)),
                "n_chord_slots": int(tone_mask.sum()),
                "feature_cols": "|".join(map(str, cols)),
            }
        )

        print(f"[{fs_name}] placement={placement_acc:.3f} | tone={tone_acc:.3f} | n_features={len(cols)}")

        if save_models:
            out_dir = cfg.paths.model_dir / f"feature_set_{fs_name}"
            models.save(out_dir)

    df = pd.DataFrame(results)
    df = df.sort_values(by=["tone_accuracy", "placement_accuracy"], ascending=False).reset_index(drop=True)

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    best = df.iloc[0].to_dict() if len(df) else {}
    if best:
        print("\nBest by tone_accuracy then placement_accuracy:")
        print(
            f"- {best['feature_set']}: tone={best['tone_accuracy']:.3f}, "
            f"placement={best['placement_accuracy']:.3f} (n_features={best['n_features']})"
        )
    print(f"\nWrote {out_path}")

    preview_setting_id: int | None = typer.Option(None, "--preview-setting-id", help="Print head() for this setting_id."),
    preview_n: int = typer.Option(5, "--preview-n", help="Rows to print for previews."),



@app.command("harmonize-chordless")
def harmonize_chordless_cmd(config: str = DEFAULT_CONFIG) -> None:
    """Predict chords for chordless settings and write a copy/pasteable ABC CSV."""
    out_csv = harmonize_chordless(config)
    print(f"Wrote interpolated chordless CSV -> {out_csv}")

    preview_setting_id: int | None = typer.Option(None, "--preview-setting-id", help="Print head() for this setting_id."),
    preview_n: int = typer.Option(5, "--preview-n", help="Rows to print for previews."),

