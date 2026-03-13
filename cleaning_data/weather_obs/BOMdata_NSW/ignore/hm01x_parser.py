#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

COLUMN_NAMES = [
    "record_id",          # 0
    "blank1",             # 1
    "station_number",     # 2
    "blank2",             # 3
    "datetime_local_raw", # 4
    "blank3",             # 5
    "datetime_utc_raw",   # 6
    "blank4",             # 7
    "rain_since_9am",     # 8
    "rain_qc",            # 9
    "temp",               # 10
    "temp_qc",            # 11
    "dewpoint",           # 12
    "dewpoint_qc",        # 13
    "rh",                 # 14
    "rh_qc",              # 15
    "wind_speed",         # 16
    "wind_speed_qc",      # 17
    "wind_dir",           # 18
    "wind_dir_qc",        # 19
    "wind_gust",          # 20
    "wind_gust_qc",       # 21
    "aws_flag",           # 22
    "blank5",             # 23
    "end_marker"          # 24
]

def parse_hm01x_file(filepath: Path) -> pd.DataFrame:
    try:
        df = pd.read_fwf(filepath, header=None, names=COLUMN_NAMES)
    except Exception:
        return None

    df["datetime_local"] = pd.to_datetime(
        df["datetime_local_raw"], format="%d/%m/%Y %H:%M", errors="coerce"
    )

    df = df.dropna(subset=["datetime_local"])

    df = df[[
        "datetime_local",
        "rain_since_9am", "rain_qc",
        "temp", "temp_qc",
        "dewpoint", "dewpoint_qc",
        "rh", "rh_qc",
        "wind_speed", "wind_speed_qc",
        "wind_dir", "wind_dir_qc",
        "wind_gust", "wind_gust_qc",
        "aws_flag"
    ]]

    return df
