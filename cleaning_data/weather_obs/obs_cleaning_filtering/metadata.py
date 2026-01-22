# metadata.py

import pandas as pd
from pathlib import Path

# Path to the Excel metadata file
ALL_STATIONS_PATH = (
    "/home/565/pv3484/aus_substation_electricity/data/raw_data/"
    "All_weatherstations_information.xlsx"
)


def _clean_columns(df):
    """
    Standardize column names:
    - strip whitespace
    - lowercase
    - replace spaces and slashes with underscores
    - remove parentheses and '#_'
    """
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("/", "_")
        .str.replace("(", "")
        .str.replace(")", "")
        .str.replace("#_", "")
    )
    return df


def load_station_metadata(stndet_path):
    """
    Load StnDet metadata (no header) and merge with Excel metadata.
    Ensures:
        - station_id is consistent
        - lat/lon/name always present when available
        - Excel-only stations are included (outer join)
    """

    # ---------------------------------------------------------
    # Load StnDet (no header)
    # ---------------------------------------------------------
    stndet = pd.read_csv(stndet_path, header=None, dtype=str)

    stndet.columns = [
        "record_type",
        "station_id",
        "district_code",
        "name",
        "opened",
        "closed",
        "lat",
        "lon",
        "elev",
        "state",
        "col10",
        "col11",
        "wmo",
        "start_year",
        "end_year",
        "col15",
        "col16",
        "col17",
        "col18",
        "col19",
        "col20",
        "col21",
    ]

    stndet = stndet.apply(lambda col: col.str.strip() if col.dtype == "object" else col)
    stndet = _clean_columns(stndet)

    # ---------------------------------------------------------
    # Load Excel metadata
    # ---------------------------------------------------------
    allstations = pd.read_excel(ALL_STATIONS_PATH, dtype=str)

    allstations = allstations.apply(
        lambda col: col.str.strip() if col.dtype == "object" else col
    )
    allstations = _clean_columns(allstations)

    # Standardize expected names
    allstations = allstations.rename(
        columns={
            "station_number": "station_id",
            "station_name": "name",
            "latitude": "lat",
            "longitude": "lon",
        }
    )

    # ---------------------------------------------------------
    # Merge StnDet + Excel metadata
    # OUTER JOIN ensures Excel-only stations are included
    # ---------------------------------------------------------
    merged = stndet.merge(
        allstations,
        on="station_id",
        how="outer",
        suffixes=("", "_alt"),
    )

    # Safe fallback fill for key fields
    for col in ["lat", "lon", "name"]:
        alt = f"{col}_alt"
        if alt in merged.columns:
            merged[col] = merged[col].fillna(merged[alt])

    # Drop alt columns
    merged = merged[[c for c in merged.columns if not c.endswith("_alt")]]

    return merged


def resolve_station_metadata(station_id, metadata_df):
    """
    Return metadata row as a dict for a given station_id.
    Works for:
        - StnDet stations
        - Excel-only stations
    """
    station_id = str(station_id)
    row = metadata_df.loc[metadata_df["station_id"] == station_id]

    if row.empty:
        return None

    return row.iloc[0].to_dict()