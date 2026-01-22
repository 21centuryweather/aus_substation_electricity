__file_2__: "Cleans and renames the columns"

import pandas as pd

def clean_chunk(df):
    """
    Clean VIC HD01D weather observation data.
    Builds a proper datetime column from BOM's split fields.
    """

    df = df.rename(columns={
        " Year Month Day Hour Minutes in YYYY": "year",
        "MM": "month",
        "DD": "day",
        "HH24": "hour",
        "MI format in Local time": "minute"
    })

    # Clean column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    for col in ["year", "month", "day", "hour", "minute"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["year", "month", "day", "hour", "minute"])

    df["datetime"] = pd.to_datetime(
        df[["year", "month", "day", "hour", "minute"]],
        errors="coerce"
    )

    df = df.dropna(subset=["datetime"])

    return df