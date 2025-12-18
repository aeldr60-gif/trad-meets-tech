import typer
from rich import print

from trad_chords.config import load_config
from trad_chords.io.fetch import fetch_thesession_csvs

from trad_chords.io.loaders import load_tunes_csv, load_popularity_csv


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
