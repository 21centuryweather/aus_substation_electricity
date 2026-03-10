"""
rainfall_conversion.py
Convert BOM 'precipitation since 9am' into rainfall per observation
and prepare the rainfall column for downstream resampling.

Author: Pia
"""

import pandas as pd
from .utils import compute_rainfall_per_obs


# ---------------------------------------------------------------------
# MAIN FUNCTION
# ---------------------------------------------------------------------

def add_rainfall_per_observation(df):
    """
    Add a 'rain_obs' column representing rainfall per observation (mm).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain:
        - datetime_local (sorted)
        - precip_since_9am

    Returns
    -------
    pd.DataFrame
        With new column:
        - rain_obs (mm)
    """
    if df.empty:
        df["rain_obs"] = []
        return df

    # Ensure sorted for correct diff logic
    df = df.sort_values("datetime_local").reset_index(drop=True)

    df["rain_obs"] = compute_rainfall_per_obs(df, col="precip_since_9am")

    return df


# ---------------------------------------------------------------------
# OPTIONAL: CLEAN-UP FUNCTION
# ---------------------------------------------------------------------

def drop_cumulative_rainfall(df):
    """
    Remove the 'precip_since_9am' column once rainfall per observation
    has been computed.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    if "precip_since_9am" in df.columns:
        df = df.drop(columns=["precip_since_9am"])
    return df
