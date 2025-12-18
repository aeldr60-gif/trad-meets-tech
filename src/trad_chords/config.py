from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import yaml


@dataclass(frozen=True)
class PathsConfig:
    raw_tunes_csv: Path
    raw_popularity_csv: Path
    processed_dir: Path
    outputs_dir: Path
    model_dir: Path


@dataclass(frozen=True)
class SourcesConfig:
    tunes_url: str
    popularity_url: str


@dataclass(frozen=True)
class AppConfig:
    paths: PathsConfig
    sources: SourcesConfig


def load_config(path: str | Path) -> AppConfig:
    p = Path(path)
    data: Dict[str, Any] = yaml.safe_load(p.read_text(encoding="utf-8"))

    paths = data["paths"]
    sources = data["sources"]

    return AppConfig(
        paths=PathsConfig(
            raw_tunes_csv=Path(paths["raw_tunes_csv"]),
            raw_popularity_csv=Path(paths["raw_popularity_csv"]),
            processed_dir=Path(paths["processed_dir"]),
            outputs_dir=Path(paths["outputs_dir"]),
            model_dir=Path(paths["model_dir"]),
        ),
        sources=SourcesConfig(
            tunes_url=str(sources["tunes_url"]),
            popularity_url=str(sources["popularity_url"]),
        ),
    )
