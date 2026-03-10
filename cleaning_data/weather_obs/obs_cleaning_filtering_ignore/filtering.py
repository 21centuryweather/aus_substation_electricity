__file_3__: "Filters the data to 2004 - 2018. Additional filtering can be added later (ie. lat/lon)"

import pandas as pd

# ---------------------------------------------------------
# Year range you want to keep
# ---------------------------------------------------------
DEFAULT_START_YEAR = 2004
DEFAULT_END_YEAR = 2018


# ---------------------------------------------------------
# Core filtering function
# ---------------------------------------------------------
def filter_by_year(df, start_year=DEFAULT_START_YEAR, end_year=DEFAULT_END_YEAR):
    """
    Filter a cleaned DataFrame to keep only rows whose datetime
    falls between start_year and end_year (inclusive).

    Assumes:
        - df contains a valid 'datetime' column
        - cleaning.py has already run
    """

    if "datetime" not in df.columns:
        # If datetime is missing, return empty — this is safer than guessing
        return df.iloc[0:0]

    years = df["datetime"].dt.year
    mask = (years >= start_year) & (years <= end_year)

    return df.loc[mask]