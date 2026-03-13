#!/usr/bin/env python3
from pathlib import Path
from build_resampled_csv import build_resampled_master

def main():
    raw_dir = Path("/home/565/pv3484/aus_substation_electricity/data/BOM_NSW_weather_processed_v2/raw_pipeline")
    output_dir = Path("/home/565/pv3484/aus_substation_electricity/data/BOM_NSW_weather_processed_v2/resampled_pipeline")
    master_csv = output_dir / "BOM_NSW_weather_master_resampled.csv"
    stndet_path = Path("/home/565/pv3484/aus_substation_electricity/data/BOMdata_NSW/HM01X_StnDet_568718610984669.txt")

    print("Running 30‑minute RESAMPLED pipeline...")
    print(f"Raw station CSVs: {raw_dir}")
    print(f"Output dir:       {output_dir}")
    print(f"Master CSV:       {master_csv}")

    build_resampled_master(raw_dir, output_dir, master_csv, stndet_path)

    print("Resampled pipeline complete.")
    print(f"Resampled master CSV written to: {master_csv}")

if __name__ == "__main__":
    main()
