__file_2__: "Cleans and renames the columns"


import pandas as pd

def clean_chunk(df):
    """
    Clean VIC HD01D weather observation data.
    Builds a proper datetime column from BOM's split fields.
    """

    # Rename columns to simpler names (note the leading space!)
    df = df.rename(columns={
        " Year Month Day Hour Minutes in YYYY": "year",
        "MM": "month",
        "DD": "day",
        "HH24": "hour",
        "MI format in Local time": "minute"
    })

    # Convert to numeric
    for col in ["year", "month", "day", "hour", "minute"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with missing datetime components
    df = df.dropna(subset=["year", "month", "day", "hour", "minute"])

    # Build datetime
    df["datetime"] = pd.to_datetime(
        df[["year", "month", "day", "hour", "minute"]],
        errors="coerce"
    )

    # Drop rows where datetime failed
    df = df.dropna(subset=["datetime"])

    return df