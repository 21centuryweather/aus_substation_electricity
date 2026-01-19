__title__: "Isolate and Move Weather Station Data"
__purpose__: "Isolate weather stations within Greater Melbourne and Greater Brisbane (meotropolitan areas) and move into a different folder"
__author__: "Pia Vassallo"

import os
import zipfile
import shutil
import pandas as pd

# ---------------------------------------------------------
# Load metro station IDs
# ---------------------------------------------------------
def load_metro_station_ids(station_info_path):
    df = pd.read_excel(station_info_path)
    df["Station_number"] = df["Station_number"].astype(str)

    melb = df[
        (df["Latitude"].between(-38.5, -37.4)) &
        (df["Longitude"].between(144.4, 145.6))
    ]

    bris = df[
        (df["Latitude"].between(-28.4, -27.0)) &
        (df["Longitude"].between(152.5, 153.5))
    ]

    return set(melb["Station_number"]), set(bris["Station_number"])


# ---------------------------------------------------------
# Extract + MOVE only files whose names contain metro station IDs
# ---------------------------------------------------------
def extract_and_move_matching_files(zip_dir, station_ids, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for zip_name in os.listdir(zip_dir):
        if not zip_name.endswith(".zip"):
            continue

        zip_path = os.path.join(zip_dir, zip_name)

        with zipfile.ZipFile(zip_path, "r") as z:
            for member in z.namelist():

                # Identify station ID from filename
                # Assumes filenames contain the station ID as a number
                station_in_name = any(station_id in member for station_id in station_ids)

                if station_in_name:
                    # Extract to a temp location inside the zip directory
                    temp_extract_path = z.extract(member, path=zip_dir)

                    # Move to cleaned output directory
                    dest_path = os.path.join(output_dir, os.path.basename(member))
                    shutil.move(temp_extract_path, dest_path)

                    print(f"MOVED: {member} → {dest_path}")

def extract_and_move_stndet(zip_dir, output_dir):
    """
    Extracts the single StnDet file from each ZIP in zip_dir
    and moves it into output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    for zip_name in os.listdir(zip_dir):
        if not zip_name.endswith(".zip"):
            continue

        zip_path = os.path.join(zip_dir, zip_name)

        with zipfile.ZipFile(zip_path, "r") as z:
            # find the StnDet file
            stndet_files = [f for f in z.namelist() if "StnDet" in f]

            if len(stndet_files) == 0:
                print(f"No StnDet file found in {zip_name}")
                continue

            if len(stndet_files) > 1:
                print(f"Warning: multiple StnDet files in {zip_name}")

            stndet = stndet_files[0]

            # extract to temp location
            temp_path = z.extract(stndet, path=zip_dir)

            # move to cleaned folder
            dest_path = os.path.join(output_dir, os.path.basename(stndet))
            shutil.move(temp_path, dest_path)

            print(f"MOVED StnDet: {stndet} → {dest_path}")

# ---------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------
def isolate_metro_weather_data():
    station_info_path = "/g/data/ng72/pv3484/substation_data/raw_data/All_stations_information.xlsx"

    # Raw ZIP directories
    melb_raw = "/home/565/pv3484/aus_substation_electricity/data/raw_data/GREATER_MELBOURNE_obs"
    bris_raw = "/home/565/pv3484/aus_substation_electricity/data/raw_data/GREATER_BRISBANE_obs"

    # Cleaned output directories
    melb_out = "/home/565/pv3484/aus_substation_electricity/data/cleaned_data/VIC/weather_obs"
    bris_out = "/home/565/pv3484/aus_substation_electricity/data/cleaned_data/QLD/weather_obs"

    # Load metro station IDs
    melb_ids, bris_ids = load_metro_station_ids(station_info_path)

    # Process each region
    extract_and_move_matching_files(melb_raw, melb_ids, melb_out)
    extract_and_move_matching_files(bris_raw, bris_ids, bris_out)


if __name__ == "__main__":
    isolate_metro_weather_data()