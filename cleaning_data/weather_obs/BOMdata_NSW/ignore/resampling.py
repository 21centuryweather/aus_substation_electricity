"""
resampling.py
Resample parsed + cleaned BOM HM01X weather data to 30-minute intervals,
aligned with electricity demand timestamps.

Author: Pia
"""

import pandas as pd


# ---------------------------------------------------------------------
# MAIN RESAMPLING FUNCTION
# ---------------------------------------------------------------------

def resample_to_30min(df):
    """
    Resample weather observations to 30-minute intervals.

    Aggregation rules:
    - temp: mean
    - rh: mean
    - wind_speed: mean
    - wind_dir: mean (simple mean; circular mean can be added later)
    - rain_obs: sum (rainfall per observation)

    Parameters
    ----------
    df : pd.DataFrame
        Must contain:
        - datetime_local (tz-aware)
        - temp, rh, wind_speed, wind_dir, rain_obs

    Returns
    -------
    pd.DataFrame
        Resampled to 30-minute intervals with aligned timestamps.
    """
    if df.empty:
        return df

    # Ensure sorted
    df = df.sort_values("datetime_local").reset_index(drop=True)

    # Set index for resampling
    df = df.set_index("datetime_local")

    # Define aggregation rules
    agg = {
        "temp": "mean",
        "rh": "mean",
        "wind_speed": "mean",
        "wind_dir": "mean",
        "rain_obs": "sum",
    }

    # Resample to 30-minute intervals
    out = df.resample("30T").agg(agg)

    # Restore index as column
    out = out.reset_index()

    return out


# ---------------------------------------------------------------------
# OPTIONAL: RENAME FOR DEMAND ALIGNMENT
# ---------------------------------------------------------------------

def rename_timestamp_for_demand(df, new_name="StartDeliveryTime"):
    """
    Rename the datetime column to match demand data conventions.

    Parameters
    ----------
    df : pd.DataFrame
    new_name : str
        New column name for datetime_local.

    Returns
    -------
    pd.DataFrame
    """
    return df.rename(columns={"datetime_local": new_name})
