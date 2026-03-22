from pathlib import Path
import pandas as pd


def write_station_csv(df: pd.DataFrame, out_path: Path) -> None:
    """
    Write a station DataFrame to CSV.

    Ensures the index is named 'timestamp_utc' for clarity.
    """
    if df.index.name is None:
        df = df.copy()
        df.index.name = "timestamp_utc"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path)
