"""
build_master_csv.py
Incremental, HPC-safe builder for BOM HM01X weather data.

Pipeline per station:
1. Parse each HM01X file (streaming)
2. Convert rainfall since 9am → rainfall per observation
3. Resample to 30-minute intervals
4. Write station-level CSV
5. Append to master CSV (optional)

Author: Pia
"""

import pandas as pd
from pathlib import Path

from .hm01x_parser import parse_hm01x_file
from .rainfall_conversion import add_rainfall_per_observation, drop_cumulative_rainfall
from .resampling import resample_to_30min
from .utils import ensure_dir, list_hm01x_files


# ---------------------------------------------------------------------
# PROCESS A SINGLE HM01X FILE
# ---------------------------------------------------------------------

def process_single_file(filepath):
    """
    Parse → rainfall conversion → resample a single HM01X file.

    Parameters
    ----------
    filepath : str or Path

    Returns
    -------
    pd.DataFrame
        Cleaned, resampled weather data for this file.
    """
    df = parse_hm01x_file(filepath)

    if df.empty:
        return df

    df = add_rainfall_per_observation(df)
    df = drop_cumulative_rainfall(df)

    df = resample_to_30min(df)

    return df


# ---------------------------------------------------------------------
# PROCESS ALL FILES FOR ONE STATION
# ---------------------------------------------------------------------

def process_station_directory(station_dir, output_dir):
    """
    Process all HM01X files for a single station directory.

    Parameters
    ----------
    station_dir : str or Path
        Directory containing HM01X_Data_* files.
    output_dir : str or Path
        Directory where station-level CSV will be written.

    Returns
    -------
    Path
        Path to the station-level CSV.
    """
    station_dir = Path(station_dir)
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    files = list_hm01x_files(station_dir)
    if not files:
        return None

    station_id = files[0].name.split("_")[-1].split(".")[0]  # last numeric chunk
    out_path = output_dir / f"weather_station_{station_id}.csv"

    # Process each file incrementally
    chunks = []
    for f in files:
        df = process_single_file(f)
        if not df.empty:
            chunks.append(df)

    if not chunks:
        return None

    # Concatenate all processed files for this station
    station_df = pd.concat(chunks, ignore_index=True)

    # Sort and drop duplicates
    station_df = station_df.sort_values("datetime_local").drop_duplicates(subset=["datetime_local"])

    # Write station-level CSV
    station_df.to_csv(out_path, index=False)

    return out_path


# ---------------------------------------------------------------------
# APPEND TO MASTER CSV (OPTIONAL)
# ---------------------------------------------------------------------

def append_to_master(station_csv, master_csv):
    """
    Append a station-level CSV to the master CSV without loading
    the entire master file into memory.

    Parameters
    ----------
    station_csv : str or Path
    master_csv : str or Path
    """
    station_csv = Path(station_csv)
    master_csv = Path(master_csv)

    # If master doesn't exist, copy headers + data
    if not master_csv.exists():
        station_df = pd.read_csv(station_csv)
        station_df.to_csv(master_csv, index=False)
        return

    # Append without headers
    station_df = pd.read_csv(station_csv)
    station_df.to_csv(master_csv, mode="a", header=False, index=False)


# ---------------------------------------------------------------------
# PROCESS ALL STATIONS
# ---------------------------------------------------------------------

def process_all_stations(root_dir, output_dir, master_csv=None):
    """
    Process all station directories under a root folder.

    Parameters
    ----------
    root_dir : str or Path
        Folder containing multiple station directories.
    output_dir : str or Path
        Where station-level CSVs will be written.
    master_csv : str or Path or None
        If provided, append each station's output to this file.

    Returns
    -------
    list[Path]
        Paths to all station-level CSVs.
    """
    root_dir = Path(root_dir)
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    station_dirs = [d for d in root_dir.iterdir() if d.is_dir()]

    outputs = []

    for sd in station_dirs:
        station_csv = process_station_directory(sd, output_dir)
        if station_csv:
            outputs.append(station_csv)
            if master_csv:
                append_to_master(station_csv, master_csv)

    return outputs
d