from pathlib import Path
import pandas as pd

from parse_hm01x import parse_hm01x_file
from rainfall import convert_cumulative_rainfall
from resample import resample_half_hourly
from metadata import load_metadata, get_station_metadata
from writer import write_station_csv

# Absolute paths
DATA_ROOT = Path("/home/565/pv3484/aus_substation_electricity/data")

RAW_ROOT = DATA_ROOT / "BOMdata_NSW"
OUT_ROOT = DATA_ROOT / "BOM_NSW_weather_processed_v3"
META_FILE = RAW_ROOT / "HM01X_StnDet_568718610984669.txt"


def process_all_stations():
    OUT_ROOT.mkdir(exist_ok=True)

    # Load metadata once
    metadata = load_metadata(META_FILE)

    # Loop through station directories
    for station_dir in sorted(d for d in RAW_ROOT.iterdir() if d.is_dir()):

        # Skip hidden folders like .ipynb_checkpoints
        if station_dir.name.startswith("."):
            continue

        station_id = station_dir.name
        out_file = OUT_ROOT / f"weather_station_{station_id}.csv"

        print(f"Processing station {station_id}...")

        # Skip if already processed
        if out_file.exists():
            print(f"  Skipping {station_id} (already processed)")
            continue

        try:
            # Find HM01X files inside the station folder
            data_files = sorted(station_dir.glob("HM01X_Data_*.txt"))
            if not data_files:
                print(f"  No HM01X data files found in {station_dir}")
                continue

            # Parse and combine all files
            dfs = []
            for f in data_files:
                df = parse_hm01x_file(f)
                dfs.append(df)

            df = pd.concat(dfs).sort_index()

            # Convert rainfall
            df = convert_cumulative_rainfall(df)

            # Resample to 30-minute intervals
            df = resample_half_hourly(df)

            # Attach metadata
            meta = get_station_metadata(metadata, station_id)
            df["lat"] = meta["lat"]
            df["lon"] = meta["lon"]
            df["station_name"] = meta["name"]
            df["station_id"] = station_id

            # Write output
            write_station_csv(df, out_file)
            print(f"  Saved {out_file}")

        except Exception as e:
            print(f"  Error processing {station_id}: {e}")
            continue


if __name__ == "__main__":
    process_all_stations()
