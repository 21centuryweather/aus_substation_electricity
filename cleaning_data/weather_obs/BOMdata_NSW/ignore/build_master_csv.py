#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from hm01x_parser import parse_hm01x_file
from station_metadata import load_station_metadata

def process_single_file(filepath: Path) -> pd.DataFrame:
    df = parse_hm01x_file(filepath)
    return df if df is not None and not df.empty else pd.DataFrame()

def process_station_directory(station_dir: Path, output_dir: Path, metadata: pd.DataFrame) -> Path:
    station_id = station_dir.name
    output_csv = output_dir / f"weather_station_{station_id}.csv"

    data_files = sorted(station_dir.glob("HM01X_Data_*.txt"))
    if not data_files:
        return output_csv

    dfs = [df for f in data_files if not (df := process_single_file(f)).empty]
    if not dfs:
        return output_csv

    station_df = pd.concat(dfs, ignore_index=True)
    station_df["station_id"] = station_id

    # merge lat/lon
    station_df = station_df.merge(metadata, on="station_id", how="left")

    output_dir.mkdir(parents=True, exist_ok=True)
    station_df.to_csv(output_csv, index=False)

    return output_csv

def build_master_csv(output_dir: Path, master_csv: Path):
    csv_files = sorted(output_dir.glob("weather_station_*.csv"))
    if not csv_files:
        return

    dfs = [pd.read_csv(f) for f in csv_files]
    master_df = pd.concat(dfs, ignore_index=True)
    master_df.to_csv(master_csv, index=False)

def process_all_stations(raw_root: Path, output_dir: Path, master_csv: Path, stndet_path: Path):
    metadata = load_station_metadata(stndet_path)

    station_dirs = sorted(
        d for d in raw_root.iterdir()
        if d.is_dir() and d.name.isdigit()
    )

    for sd in tqdm(station_dirs, desc="Processing stations"):
        process_station_directory(sd, output_dir, metadata)

    build_master_csv(output_dir, master_csv)
