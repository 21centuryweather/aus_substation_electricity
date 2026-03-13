import pandas as pd


def resample_half_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample to 30-minute intervals.

    - temp, dewpoint, relative_humidity, wind_speed_kmh, wind_direction_deg: mean
    - precip_incremental: sum
    """
    if df.empty:
        return df

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex for resampling.")

    agg_map = {}

    for col in ["temp", "dewpoint", "relative_humidity",
                "wind_speed_kmh", "wind_direction_deg"]:
        if col in df.columns:
            agg_map[col] = "mean"

    if "precip_incremental" in df.columns:
        agg_map["precip_incremental"] = "sum"

    # If no known columns, just return as-is
    if not agg_map:
        return df

    df_resampled = df.resample("30T").agg(agg_map)

    return df_resampled
