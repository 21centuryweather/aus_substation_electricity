from build_master_csv import process_all_stations

ROOT_DIR = "/home/565/pv3484/aus_substation_electricity/data/BOMdata_NSW"
OUTPUT_DIR = "/home/565/pv3484/aus_substation_electricity/data/BOM_NSW_weather_processed"
MASTER_CSV = "/home/565/pv3484/aus_substation_electricity/data/cleaned_data/BOM_NSW_weather_master.csv"

if __name__ == "__main__":
    process_all_stations(ROOT_DIR, OUTPUT_DIR, MASTER_CSV)
    print("Weather processing complete.")
