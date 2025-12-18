from pathlib import Path
import pandas as pd


def load_tunes_csv(path: Path) -> pd.DataFrame:
    """
    Load TheSession csv/tunes.csv.

    This is a comma-separated CSV with a multiline quoted ABC field.
    We use the python engine and skip malformed lines.
    """
    return pd.read_csv(
        path,
        sep=",",
        engine="python",
        quotechar='"',
        on_bad_lines="skip",
    )


def load_popularity_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)