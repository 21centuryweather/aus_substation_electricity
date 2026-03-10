"""
hm01x_parser.py
Streaming parser for BOM HM01X weather observation files.

Reads each file line-by-line (HPC-safe), extracts only the variables
you need, applies quality flags, parses timestamps, and prepares the
dataframe for rainfall conversion and resampling.

Author: Pia
"""

import pandas as pd
from pathlib import Path

from utils import (
    parse_local_timestamp,
    apply_quality_mask,
)


# ---------------------------------------------------------------------
# BYTE POSITIONS (from BOM notes)
# ---------------------------------------------------------------------
# These are 1-based positions; Python slicing is 0-based and end-exclusive.

COLS = {
    "station_id":      (3, 9),     # bytes 4-9
    "ts_local":        (10, 26),   # bytes 11-26
    "precip":          (44, 50),   # bytes 45-50
    "q_precip":        (51, 52),   # byte 52
    "temp":            (53, 58),   # bytes 54-58
    "q_temp":          (59, 60),   # byte 60
    "rh":              (69, 72),   # bytes 70-72
    "q_rh":            (73, 74),   # byte 74
    "wind_speed":      (75, 80),   # bytes 76-80
    "q_wind":          (81, 82),   # byte 82
    "wind_dir":        (83, 86),   # bytes 84-86
    "q_wind_dir":      (87, 88),   # byte 88
}


# ---------------------------------------------------------------------
# PARSING HELPERS
# ---------------------------------------------------------------------

def _extract(line, start, end):
    """Extract substring from fixed-width line."""
    return line[start:end].strip()


def _to_float(val):
    """Convert string to float safely."""
    try:
        return float(val)
    except:
        return pd.NA


# ---------------------------------------------------------------------
# MAIN PARSER
# ---------------------------------------------------------------------

def parse_hm01x_file(filepath):
    """
    Parse a single HM01X file into a tidy dataframe.

    Parameters
    ----------
    filepath : str or Path

    Returns
    -------
    pd.DataFrame
        Columns:
        - station_id
        - datetime_local (tz-aware)
        - temp
        - rh
        - wind_speed
        - wind_dir
        - precip_since_9am
        - q_temp, q_rh, q_wind, q_precip, q_wind_dir
    """
    filepath = Path(filepath)

    rows = []

    with filepath.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("hm"):
                continue  # skip non-data lines

            # Extract fields
            station_id = _extract(line, *COLS["station_id"])
            ts_local   = _extract(line, *COLS["ts_local"])

            temp       = _to_float(_extract(line, *COLS["temp"]))
            q_temp     = _extract(line, *COLS["q_temp"])

            rh         = _to_float(_extract(line, *COLS["rh"]))
            q_rh       = _extract(line, *COLS["q_rh"])

            wind_speed = _to_float(_extract(line, *COLS["wind_speed"]))
            q_wind     = _extract(line, *COLS["q_wind"])

            wind_dir   = _to_float(_extract(line, *COLS["wind_dir"]))
            q_wind_dir = _extract(line, *COLS["q_wind_dir"])

            precip     = _to_float(_extract(line, *COLS["precip"]))
            q_precip   = _extract(line, *COLS["q_precip"])

            # Parse timestamp
            dt_local = parse_local_timestamp(ts_local)

            rows.append({
                "station_id": station_id,
                "datetime_local": dt_local,

                "temp": temp,
                "q_temp": q_temp,

                "rh": rh,
                "q_rh": q_rh,

                "wind_speed": wind_speed,
                "q_wind": q_wind,

                "wind_dir": wind_dir,
                "q_wind_dir": q_wind_dir,

                "precip_since_9am": precip,
                "q_precip": q_precip,
            })

    df = pd.DataFrame(rows)

    # Drop rows with invalid timestamps
    df = df.dropna(subset=["datetime_local"])

    # Sort for rainfall conversion + resampling
    df = df.sort_values("datetime_local")

    # Apply quality masks
    df["temp"]       = apply_quality_mask(df["temp"], df["q_temp"])
    df["rh"]         = apply_quality_mask(df["rh"], df["q_rh"])
    df["wind_speed"] = apply_quality_mask(df["wind_speed"], df["q_wind"])
    df["wind_dir"]   = apply_quality_mask(df["wind_dir"], df["q_wind_dir"])
    df["precip_since_9am"] = apply_quality_mask(df["precip_since_9am"], df["q_precip"])

    return df.reset_index(drop=True)