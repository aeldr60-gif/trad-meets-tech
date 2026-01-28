from pathlib import Path
from urllib.request import Request, urlopen

"""
Simple helpers for downloading TheSession CSV sources.

`download_file()` wraps a basic URL fetch with a custom User Agent and ensures
the destination directory exists. `fetch_thesession_csvs()` downloads the tunes
and popularity CSV files from the configured URLs, raising an error if either
URL is missing.
"""


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": "trad-meets-tech/0.1"})
    with urlopen(req) as resp:
        dest.write_bytes(resp.read())


def fetch_thesession_csvs(
    tunes_url: str,
    popularity_url: str,
    tunes_dest: Path,
    popularity_dest: Path,
) -> None:
    if not tunes_url or not popularity_url:
        raise ValueError("Missing `sources.tunes_url` or `sources.popularity_url` in configs/default.yaml")
    download_file(tunes_url, tunes_dest)
    download_file(popularity_url, popularity_dest)