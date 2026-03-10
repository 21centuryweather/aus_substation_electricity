"""
utils.py
Lightweight helper utilities for the BOM HM01X weather-processing pipeline.

These functions are intentionally minimal and dependency-light so they can be
safely imported across HPC jobs without memory overhead.

Author: Pia
"""

from pathlib import Path
import pandas as pd
import pytz


# ---------------------------------------------------------------------
# DIRECTORY + FILE HELPERS
# ---------------------------------------------------------------------

def ensure_dir(path):
    """
    Create a directory if it does not already exist.
    Safe to call repeatedly.

    Parameters
    ----------
    path : str or Path
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def list_hm01x_files(directory):
    """
    Return a sorted list of HM01X data files in a directory.

    Parameters
    ----------
    directory : str or Path

    Returns
    -------
    list[Path]
    """
    directory = Path(directory)
    return sorted([f for f in directory.iterdir() if f.name.startswith("HM01X_Data")])


# ---------------------------------------------------------------------
# TIME HELPERS
# ---------------------------------------------------------------------

_SYD_TZ = pytz.timezone("Australia/Sydney")


def parse_local_timestamp(ts_str):
    """
    Parse a DD/MM/YYYY HH:MM timestamp string into a timezone-aware
    datetime in Australia/Sydney.

    Parameters
    ----------
    ts_str : str

    Returns
    -------
    pd.Timestamp (tz-aware)
    """
    dt = pd.to_datetime(ts_str, format="%d/%m/%Y %H:%M", errors="coerce")
    if pd.isna(dt):
        return pd.NaT
    return _SYD_TZ.localize(dt)


def floor_to_30min(ts):
    """
    Floor a timestamp to the nearest 30-minute boundary.

    Parameters
    ----------
    ts : pd.Timestamp

    Returns
    -------
    pd.Timestamp
    """
    if pd.isna(ts):
        return pd.NaT
    return ts.floor("30T")


# ---------------------------------------------------------------------
# RAINFALL HELPERS
# ---------------------------------------------------------------------

def compute_rainfall_per_obs(df, col="precip_since_9am"):
    """
    Convert BOM 'precipitation since 9am' into rainfall per observation.

    Logic:
    - Take diff of cumulative rainfall.
    - Negative diffs indicate the 9am reset → set to cumulative value.
    - First observation of the day uses the cumulative value directly.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain:
        - datetime_local (sorted)
        - precip_since_9am (float)
    col : str
        Column name for cumulative rainfall.

    Returns
    -------
    pd.Series
        Rainfall per observation (mm)
    """
    # Difference between consecutive observations
    diff = df[col].diff()

    # Reset at 9am: negative diffs indicate the daily reset
    diff = diff.clip(lower=0)

    # First observation after 9am reset
    is_9am = (
        df["datetime_local"].dt.hour.eq(9)
        & df["datetime_local"].dt.minute.eq(0)
    )
    diff[is_9am] = df.loc[is_9am, col]

    return diff.fillna(0)


# ---------------------------------------------------------------------
# QUALITY HELPERS
# ---------------------------------------------------------------------

def apply_quality_mask(series, qflag, allowed=("Y", "N")):
    """
    Mask values based on BOM quality flags.

    Parameters
    ----------
    series : pd.Series
        Data values.
    qflag : pd.Series
        Quality flag column.
    allowed : tuple
        Flags to keep (default: Y and N).

    Returns
    -------
    pd.Series
        Series with disallowed values replaced by NaN.
    """
    mask = qflag.isin(allowed)
    return series.where(mask, other=pd.NA)
