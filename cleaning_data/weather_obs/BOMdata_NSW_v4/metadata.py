from pathlib import Path
import pandas as pd


def load_metadata(path: Path) -> pd.DataFrame:
    """
    Load HM01X station details file (StnDet) into a DataFrame.

    Uses the layout from the notes file.
    """
    cols = [
        "rec",
        "station_id",
        "district_code",
        "name",
        "opened",
        "closed",
        "lat",
        "lon",
        "method",
        "state",
        "elev_m",
        "baro_elev_m",
        "wmo_index",
        "first_year",
        "last_year",
        "pct_complete",
        "pct_Y",
        "pct_N",
        "pct_W",
        "pct_S",
        "pct_I",
        "end",
    ]

    df = pd.read_csv(
        path,
        header=None,
        names=cols,
        dtype={"station_id": str},
    )

    # Normalise station_id to 6 digits
    df["station_id"] = df["station_id"].astype(str).str.zfill(6)

    # Coerce numeric fields
    for c in ["lat", "lon", "elev_m", "baro_elev_m"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def get_station_metadata(metadata: pd.DataFrame, station_id: str) -> dict:
    """
    Return a dict of metadata for a given station_id.
    """
    sid = str(station_id).zfill(6)
    row = metadata.loc[metadata["station_id"] == sid]

    if row.empty:
        raise ValueError(f"No metadata found for station_id {sid}")

    r = row.iloc[0]

    return {
        "station_id": sid,
        "name": r["name"],
        "lat": r["lat"],
        "lon": r["lon"],
        "elev_m": r["elev_m"],
        "state": r["state"],
    }
