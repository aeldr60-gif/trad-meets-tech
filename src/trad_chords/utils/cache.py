from pathlib import Path

def should_skip(output_path: Path, overwrite: bool) -> bool:
    return output_path.exists() and (not overwrite)
