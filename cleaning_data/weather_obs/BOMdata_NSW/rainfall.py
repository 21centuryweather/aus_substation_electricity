import pandas as pd


def convert_cumulative_rainfall(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert cumulative precipitation since 9am to incremental rainfall.

    Assumes column 'precip_cumulative' in mm.
    Returns df with an extra column 'precip_incremental'.
    """
    if "precip_cumulative" not in df.columns:
        df["precip_incremental"] = pd.NA
        return df

    # Sort by time just in case
    df = df.sort_index()

    # Simple diff
    inc = df["precip_cumulative"].diff()

    # When cumulative resets (e.g. at 9am), diff becomes negative.
    # In that case, treat the new cumulative value as the increment.
    reset_mask = (inc < 0) | inc.isna()
    inc[reset_mask] = df["precip_cumulative"][reset_mask]

    df["precip_incremental"] = inc

    return df
