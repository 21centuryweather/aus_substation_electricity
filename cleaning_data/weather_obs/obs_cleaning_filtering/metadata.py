import pandas as pd
from pathlib import Path

import pandas as pd

def load_station_metadata(stndet_path):
    """
    Load StnDet metadata which has NO header row.
    Assign correct column names and create station_id.
    """

    # Load with no header
    df = pd.read_csv(stndet_path, header=None, dtype=str)

    # Assign column names based on actual StnDet structure
    df.columns = [
        "record_type",      # st
        "station_id",       # 085099
        "district_code",    # 85
        "name",             # POUND CREEK
        "opened",           # 04/2007
        "closed",           # NaN
        "lat",              # -38.6297
        "lon",              # 145.8107
        "elev",             # blank
        "state",            # VIC
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
        "col21"
    ]

    # Strip whitespace
    df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)

    return df
    
ALL_STATIONS_PATH = "/home/565/pv3484/aus_substation_electricity/data/raw_data/All_stations_information.xlsx"

def resolve_station_metadata(station_id, stndet_df):
    # Try StnDet first
    row = stndet_df.loc[stndet_df["station_id"] == str(station_id)]
    if not row.empty:
        return row.iloc[0].to_dict()

    # Fallback: All_stations_information.xlsx
    allstations = pd.read_excel(ALL_STATIONS_PATH, dtype=str)

    allstations = allstations.rename(columns={
        "Station_number": "station_id",
        "Station_name": "name"
    })

    allstations = allstations.apply(
        lambda col: col.str.strip() if col.dtype == "object" else col
    )

    row2 = allstations.loc[allstations["station_id"] == str(station_id)]
    if not row2.empty:
        return row2.iloc[0].to_dict()

    return None