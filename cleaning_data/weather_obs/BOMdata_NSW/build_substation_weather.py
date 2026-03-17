from pathlib import Path
import pandas as pd

# Purpose of this script: 

"""
This script builds a complete substation‑level half‑hourly weather dataset by
expanding each processed BOM HM01X station time series to all electricity
substations listed in the substation_to_weather_mapping.csv file. It is the
required upstream step for regenerating weather_holiday_block_means_with_codes.csv
after the BOM station data is updated. Run this script only after all missing
or empty BOM station files (e.g., 66202) have been replaced or appropriate
fallback stations have been chosen.
"""


# === PATHS ===============================================================

DATA_ROOT = Path("/home/565/pv3484/aus_substation_electricity/data")

# Output from your existing BOM pipeline (per BOM station)
BOM_OUT_ROOT = DATA_ROOT / "BOM_NSW_weather_processed_v3"

# Substation → nearest BOM station mapping
SUBSTATION_MAP_FILE = (
    Path("/home/565/pv3484/aus_substation_electricity/pia_notebooks/"
         "NSW_data/relative_ranking/substation_to_weather_mapping.csv")
)

# New output: long-format weather for every substation
SUBSTATION_WEATHER_OUT = DATA_ROOT / "substation_weather_half_hourly.csv"


# === HELPERS =============================================================

def load_bom_station(station_id: str) -> pd.DataFrame:
    """
    Load a processed BOM station CSV produced by run_pipeline.py.
    """
    station_id = str(station_id).zfill(6)  # <-- 6 digits, not 5
    path = BOM_OUT_ROOT / f"weather_station_{station_id}.csv"

    if not path.exists():
        raise FileNotFoundError(f"No processed BOM file for station_id={station_id}")

    df = pd.read_csv(path, parse_dates=["timestamp_utc"], index_col="timestamp_utc")
    return df


# === MAIN BUILD FUNCTION ================================================

def build_substation_weather():
    print("\n=== Building substation-level weather dataset ===\n")

    # Load mapping: substation name, lat, lon, nearest_station_id, distance_km
    mapping = pd.read_csv(SUBSTATION_MAP_FILE)

    # Ensure nearest_station_id is string
    mapping["nearest_station_id"] = mapping["nearest_station_id"].astype(str)

    all_frames = []
    bom_cache: dict[str, pd.DataFrame] = {}

    for _, row in mapping.iterrows():
        sub_name = row["name"]                 # e.g. "Blakehurst"
        sub_lat = row["lat"]
        sub_lon = row["lon"]
        nearest_id = row["nearest_station_id"] # e.g. "66037"
        dist_km = row["distance_km"]

        # Load BOM station data (cached)
        if nearest_id not in bom_cache:
            try:
                bom_cache[nearest_id] = load_bom_station(nearest_id)
            except FileNotFoundError:
                print(f"WARNING: No BOM data for nearest_station_id={nearest_id} "
                      f"(substation={sub_name}) — skipping")
                continue

        bom_df = bom_cache[nearest_id].copy()

        # Attach substation metadata
        bom_df["substation_name"] = sub_name
        bom_df["substation_lat"] = sub_lat
        bom_df["substation_lon"] = sub_lon
        bom_df["nearest_station_id"] = nearest_id
        bom_df["distance_km"] = dist_km

        all_frames.append(bom_df)

    if not all_frames:
        raise RuntimeError("No substation weather frames were built — check mapping and BOM outputs.")

    # Combine all substations into one long-format table
    substation_weather = pd.concat(all_frames).sort_index()

    # Ensure index name is clear
    if substation_weather.index.name is None:
        substation_weather.index.name = "timestamp_utc"

    # Save output
    SUBSTATION_WEATHER_OUT.parent.mkdir(parents=True, exist_ok=True)
    substation_weather.to_csv(SUBSTATION_WEATHER_OUT)

    print(f"\nSaved substation-level weather to:\n  {SUBSTATION_WEATHER_OUT}\n")
    print("All substations in the mapping file are now guaranteed to have weather data.")


# === ENTRYPOINT ==========================================================

if __name__ == "__main__":
    build_substation_weather()
