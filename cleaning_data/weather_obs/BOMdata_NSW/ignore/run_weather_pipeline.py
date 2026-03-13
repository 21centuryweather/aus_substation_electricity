#!/usr/bin/env python3
from pathlib import Path
from build_master_csv import process_all_stations

def main():
    raw_root = Path("/home/565/pv3484/aus_substation_electricity/data/BOMdata_NSW")
    output_dir = Path("/home/565/pv3484/aus_substation_electricity/data/BOM_NSW_weather_processed_v2/raw_pipeline")
    master_csv = output_dir / "BOM_NSW_weather_master.csv"
    stndet_path = raw_root / "HM01X_StnDet_568718610984669.txt"

    print("Running RAW weather pipeline...")
    print(f"Raw input dir:   {raw_root}")
    print(f"Output dir:      {output_dir}")
    print(f"Master CSV:      {master_csv}")

    process_all_stations(raw_root, output_dir, master_csv, stndet_path)

    print("RAW pipeline complete.")
    print(f"Master CSV written to: {master_csv}")

if __name__ == "__main__":
    main()
