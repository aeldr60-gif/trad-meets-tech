from pathlib import Path
import pandas as pd

"""
Loaders for TheSession tune and popularity CSV files.

`load_tunes_csv()` reads the tunes.csv file using the Python CSV engine, allowing
multiline quoted ABC fields and skipping malformed rows. `load_popularity_csv()`
provides a simple wrapper around `pandas.read_csv()` for the corresponding
popularity file.
"""


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