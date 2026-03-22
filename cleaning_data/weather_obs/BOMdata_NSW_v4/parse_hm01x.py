from pathlib import Path
import pandas as pd


def find_column(columns, keyword):
    """Return the first column whose name contains the keyword."""
    keyword = keyword.lower()
    for col in columns:
        if keyword in col.lower():
            return col
    return None


def parse_hm01x_file(path: Path) -> pd.DataFrame:
    """
    Parse a comma-separated HM01X file with a header row.
    Uses keyword matching to find the correct columns.
    """

    df = pd.read_csv(path, header=0, low_memory=False)

    cols = df.columns

    # Identify columns by keyword search
    utc_col = find_column(cols, "universal coordinated")
    precip_col = find_column(cols, "precipitation since 9am")
    temp_col = find_column(cols, "air temperature")
    dew_col = find_column(cols, "dew point")
    rh_col = find_column(cols, "relative humidity")
    wspd_col = find_column(cols, "wind speed")
    wdir_col = find_column(cols, "wind direction")

    if utc_col is None:
        raise KeyError("Could not find UTC timestamp column in file")

    # Parse UTC timestamp
    df["timestamp_utc"] = pd.to_datetime(
        df[utc_col],
        format="%d/%m/%Y %H:%M",
        errors="coerce",
        utc=True
    )

    df = df.dropna(subset=["timestamp_utc"])
    df = df.set_index("timestamp_utc").sort_index()

    # Build clean output
    out = pd.DataFrame(index=df.index)

    if precip_col: out["precip_cumulative"] = pd.to_numeric(df[precip_col], errors="coerce")
    if temp_col: out["temp"] = pd.to_numeric(df[temp_col], errors="coerce")
    if dew_col: out["dewpoint"] = pd.to_numeric(df[dew_col], errors="coerce")
    if rh_col: out["relative_humidity"] = pd.to_numeric(df[rh_col], errors="coerce")
    if wspd_col: out["wind_speed_kmh"] = pd.to_numeric(df[wspd_col], errors="coerce")
    if wdir_col: out["wind_direction_deg"] = pd.to_numeric(df[wdir_col], errors="coerce")

    return out
