from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
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
class PipelineConfig:
    tune_types: list[str]
    min_tunebooks: int
    top_n: Optional[int]


@dataclass(frozen=True)
class ArtifactsConfig:
    jigs_reels_index_csv: Path


@dataclass(frozen=True)
class AppConfig:
    paths: PathsConfig
    sources: SourcesConfig
    pipeline: PipelineConfig
    artifacts: ArtifactsConfig


def load_config(path: str | Path) -> AppConfig:
    p = Path(path)
    data: Dict[str, Any] = yaml.safe_load(p.read_text(encoding="utf-8"))

    paths = data["paths"]
    sources = data["sources"]
    pipeline = data.get("pipeline", {})
    artifacts = data.get("artifacts", {})

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
        pipeline=PipelineConfig(
            tune_types=[str(x).lower() for x in pipeline.get("tune_types", ["jig", "reel"])],
            min_tunebooks=int(pipeline.get("min_tunebooks", 0)),
            top_n=pipeline.get("top_n"),
        ),
        artifacts=ArtifactsConfig(
            jigs_reels_index_csv=Path(artifacts.get("jigs_reels_index_csv", "data/processed/jigs_reels_index.csv"))
        ),
    )
