import typer
from rich import print

from trad_chords.config import load_config
from trad_chords.io.fetch import fetch_thesession_csvs

from trad_chords.io.loaders import load_tunes_csv, load_popularity_csv
from trad_chords.features.index_builder import build_jigs_reels_index, write_index




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
    print(df.head(5))
