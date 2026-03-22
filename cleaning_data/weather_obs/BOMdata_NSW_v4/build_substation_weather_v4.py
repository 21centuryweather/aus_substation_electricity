#!/usr/bin/env python3
"""
Build substation-level half-hourly weather dataset (v4).

- Uses station-level v4 weather CSVs in BOM_NSW_weather_processed_v4
- Uses substation_to_weather_mapping_v4.csv for nearest-station assignment
- Handles combined station COMBINED_066214_066022:
    - temperature/precip/etc from 066214 (OBS HILL)
    - wind from 066022 (FORT DENISON)
- Outputs a single CSV:
    substation_weather_half_hourly_v4.csv
"""

from pathlib import Path
import pandas as pd


# ---------------------------------------------------------------------
# Paths (adjust only if directory structure changes)
# ---------------------------------------------------------------------

BASE_DIR = Path("/home/565/pv3484/aus_substation_electricity")
PROCESSED_WEATHER_DIR = BASE_DIR / "data" / "BOM_NSW_weather_processed_v4"

MAPPING_PATH = (
    BASE_DIR
    / "pia_notebooks"
    / "NSW_data"
    / "relative_ranking"
    / "substation_to_weather_mapping_v4.csv"
)

OUTPUT_PATH = PROCESSED_WEATHER_DIR / "substation_weather_half_hourly_v4.csv"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def load_station_data() -> dict:
    """
    Load all station-level v4 weather CSVs into a dict keyed by 6-digit station_id string.

    Expects files named:
        weather_station_<station_id>.csv
    with columns including:
        timestamp_utc, temp, dewpoint, relative_humidity,
        wind_speed_kmh, wind_direction_deg, precip_incremental,
        lat, lon, station_name, station_id
    """
    station_data = {}

    for csv_file in PROCESSED_WEATHER_DIR.glob("weather_station_*.csv"):
        try:
            df = pd.read_csv(csv_file, parse_dates=["timestamp_utc"])
        except Exception as e:
            print(f"WARNING: Failed to read {csv_file.name}: {e}")
            continue

        # Ensure station_id is a zero-padded 6-digit string
        if "station_id" in df.columns:
            sid = str(df["station_id"].iloc[0]).strip()
            sid = "".join(ch for ch in sid if ch.isdigit())
            sid = sid.zfill(6)
        else:
            # Fallback: infer from filename
            sid = csv_file.stem.replace("weather_station_", "")
            sid = sid.strip().zfill(6)

        # Clean station_name
        if "station_name" in df.columns:
            df["station_name"] = (
                df["station_name"]
                .astype(str)
                .str.strip()
                .str.replace(r"[\(\)\[\]]", "", regex=True)
            )

        station_data[sid] = df

    return station_data


def build_combined_station(df_temp: pd.DataFrame, df_wind: pd.DataFrame) -> pd.DataFrame:
    """
    Build combined station COMBINED_066214_066022:
    - df_temp: station 066214 (OBS HILL) with full variables
    - df_wind: station 066022 (FORT DENISON) providing wind only

    Returns a merged DataFrame with:
        timestamp, temp, dewpoint, relative_humidity,
        wind_speed, wind_dir, precip_incremental, lat, lon, station_name, station_id
    """
    # Standardise timestamps
    df_temp = df_temp.rename(columns={"timestamp_utc": "timestamp"})
    df_wind = df_wind.rename(
        columns={
            "timestamp_utc": "timestamp",
            "wind_speed_kmh": "wind_speed",
            "wind_direction_deg": "wind_dir",
        }
    )

    # Subset wind columns
    wind_cols = ["timestamp", "wind_speed", "wind_dir"]
    df_wind_subset = df_wind[wind_cols]

    # Merge on timestamp
    merged = df_temp.merge(df_wind_subset, on="timestamp", how="left")

    # Mark as combined station
    merged["station_id"] = "COMBINED_066214_066022"
    merged["station_name"] = "SYDNEY COMBINED (OBS HILL + FORT DENISON)"

    return merged


def load_mapping() -> pd.DataFrame:
    """
    Load substation-to-weather mapping (v4) and ensure nearest_station_id is a 6-digit string.
    """
    mapping = pd.read_csv(MAPPING_PATH, dtype={"nearest_station_id": str})

    # Clean station IDs: digits only, zero-padded to 6
    mapping["nearest_station_id"] = (
        mapping["nearest_station_id"]
        .astype(str)
        .str.replace(r"[^\d]", "", regex=True)
        .str.zfill(6)
    )

    # Clean substation names
    mapping["name"] = (
        mapping["name"]
        .astype(str)
        .str.strip()
        .str.replace(r"[\(\)\[\]]", "", regex=True)
    )

    return mapping


# ---------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------

def build_substation_weather():
    print("\n=== Building substation-level weather dataset (v4) ===\n")

    # 1. Load station-level data
    station_data = load_station_data()

    # 2. Build combined station and inject into station_data
    sid_temp = "066214"
    sid_wind = "066022"

    if sid_temp not in station_data:
        raise RuntimeError(f"Expected station {sid_temp} not found in station_data")
    if sid_wind not in station_data:
        raise RuntimeError(f"Expected station {sid_wind} not found in station_data")

    combined_df = build_combined_station(
        station_data[sid_temp].copy(),
        station_data[sid_wind].copy(),
    )
    station_data["COMBINED_066214_066022"] = combined_df

    # 3. Load mapping
    mapping = load_mapping()

    # 4. Build substation-level weather
    all_substation_frames = []

    for _, row in mapping.iterrows():
        sub_name = row["name"]
        station_id = str(row["nearest_station_id"]).strip()

        # Special case: if mapping uses the combined station label directly
        if station_id == "COMBINED_066214_066022":
            sid_key = "COMBINED_066214_066022"
        else:
            sid_key = station_id.zfill(6)

        if sid_key not in station_data:
            print(
                f"WARNING: No BOM data for nearest_station_id={station_id} "
                f"(substation={sub_name}) - skipping"
            )
            continue

        df_station = station_data[sid_key].copy()

        # Standardise timestamp column name for output
        if "timestamp_utc" in df_station.columns:
            df_station = df_station.rename(columns={"timestamp_utc": "timestamp"})

        # Attach substation name
        df_station["substation_name"] = sub_name

        all_substation_frames.append(df_station)

    if not all_substation_frames:
        raise RuntimeError("No substation weather data could be built. Check mapping and station_data.")

    result = pd.concat(all_substation_frames, ignore_index=True)

    # 5. Sort and save
    if "timestamp" in result.columns:
        result = result.sort_values(["substation_name", "timestamp"])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)

    print("\nSaved substation-level weather to:")
    print(str(OUTPUT_PATH))
    print("\nDone.\n")


if __name__ == "__main__":
    build_substation_weather()
