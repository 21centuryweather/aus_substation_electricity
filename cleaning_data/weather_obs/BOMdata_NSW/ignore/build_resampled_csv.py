#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from rainfall_conversion import convert_rainfall  # your existing logic
from resampling import resample_to_30min          # your existing logic (assumed)
from station_metadata import load_station_metadata

def process_station_resampled(
    raw_station_csv: Path,
    metadata: pd.DataFrame,
    output_dir: Path
) -> Path:
    df = pd.read_csv(raw_station_csv, parse_dates=["datetime_local"])

    # ensure sorted
    df = df.sort_values("datetime_local")

    # rainfall conversion (using your existing function)
    df = convert_rainfall(df)

    # resample to 30‑minute (using your existing function)
    df_30 = resample_to_30min(df)

    # keep metadata (station_id, lat, lon, etc.)
    if "station_id" in df_30.columns:
        df_30 = df_30.merge(
            metadata,
            on="station_id",
            how="left",
            suffixes=("", "_meta")
        )

    station_id = df_30["station_id"].iloc[0]
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"weather_station_{station_id}_resampled.csv"
    df_30.to_csv(out_path, index=False)
    return out_path

def build_resampled_master(
    raw_dir: Path,
    output_dir: Path,
    master_csv: Path,
    stndet_path: Path
):
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    master_csv = Path(master_csv)

    metadata = load_station_metadata(stndet_path)

    raw_station_files = sorted(raw_dir.glob("weather_station_*.csv"))
    if not raw_station_files:
        print("No raw station CSVs found.")
        return

    resampled_paths = []
    for f in tqdm(raw_station_files, desc="Resampling stations"):
        out = process_station_resampled(f, metadata, output_dir)
        resampled_paths.append(out)

    dfs = [pd.read_csv(p, parse_dates=["datetime_local"]) for p in resampled_paths]
    master_df = pd.concat(dfs, ignore_index=True)
    master_df.to_csv(master_csv, index=False)
