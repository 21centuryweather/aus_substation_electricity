#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

def load_station_metadata(stndet_path: Path) -> pd.DataFrame:
    df = pd.read_fwf(stndet_path, skiprows=3)

    df = df.rename(columns={
        df.columns[0]: "station_id",
        df.columns[1]: "station_name",
        df.columns[2]: "lat",
        df.columns[3]: "lon",
        df.columns[4]: "elev"
    })

    df["station_id"] = df["station_id"].astype(str).str.zfill(6)

    return df[["station_id", "station_name", "lat", "lon", "elev"]]
