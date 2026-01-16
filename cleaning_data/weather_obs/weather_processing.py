from pathlib import Path
import pandas as pd
import zipfile
import logging
from tqdm import tqdm


# ------------------------------------------------------------
# LOGGING SETUP
# ------------------------------------------------------------

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# You can configure handlers in your notebook or main script:
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ------------------------------------------------------------
# 1. ZIP EXTRACTION
# ------------------------------------------------------------

def extract_all_zips(region_path: Path):
    """
    Extract all ZIP files inside a region directory.
    Returns a list of extracted directory paths.
    """
    zip_files = list(region_path.glob("*.zip"))
    extracted_dirs = []

    logger.info(f"Found {len(zip_files)} ZIP files in {region_path}")

    for z in tqdm(zip_files, desc="Extracting ZIPs"):
        out_dir = z.with_suffix("")
        if not out_dir.exists():
            logger.info(f"Extracting {z.name} → {out_dir}")
            with zipfile.ZipFile(z, "r") as zip_ref:
                zip_ref.extractall(out_dir)
        else:
            logger.info(f"Already extracted: {z.name}")
        extracted_dirs.append(out_dir)

    return extracted_dirs


# ------------------------------------------------------------
# 2. LOAD STATION METADATA
# ------------------------------------------------------------

STATION_COLS = [
    "record_type", "site_id", "site_num", "site_name", "start_date",
    "lat", "lon", "location_method", "state", "elev", "bar_ht", "wmo_id",
    "start_year", "end_year", "percent_complete", "percent_quality",
    "has_rain", "has_temp", "unknown1", "has_wind", "unknown2", "end_marker"
]


def load_station_metadata(extracted_dirs):
    """
    Load all station metadata files from extracted ZIP directories.
    """
    frames = []

    logger.info("Loading station metadata files")

    for d in tqdm(extracted_dirs, desc="Reading station metadata"):
        stn_files = list(d.glob("HD01D_StnDet_*.txt"))
        if not stn_files:
            logger.warning(f"No station metadata file found in {d}")
            continue

        logger.info(f"Reading station metadata: {stn_files[0].name}")

        df = pd.read_csv(
            stn_files[0],
            sep=",",
            header=None,
            skiprows=1,
            engine="python"
        )
        frames.append(df)

    stations = pd.concat(frames, ignore_index=True)
    stations.columns = STATION_COLS

    logger.info(f"Loaded {len(stations)} station metadata rows")

    return stations


# ------------------------------------------------------------
# 3. FILTER TO METRO STATIONS
# ------------------------------------------------------------

def filter_metro(stations, bbox):
    """
    Filter stations to those inside a bounding box.
    bbox = (lat_min, lat_max, lon_min, lon_max)
    """
    lat_min, lat_max, lon_min, lon_max = bbox

    logger.info("Filtering stations to metro bounding box")

    metro = stations[
        stations["lat"].astype(float).between(lat_min, lat_max)
        & stations["lon"].astype(float).between(lon_min, lon_max)
    ]

    logger.info(f"Selected {len(metro)} metro stations")

    return metro


# ------------------------------------------------------------
# 4. LOAD OBSERVATIONAL DATA FOR METRO STATIONS
# ------------------------------------------------------------

def load_metro_data(extracted_dirs, metro_ids):
    """
    Load all observational data files for the selected metro station IDs.
    """
    dfs = []

    logger.info(f"Loading observational data for {len(metro_ids)} metro stations")

    for d in extracted_dirs:
        data_files = list(d.glob("HD01D_Data_*.txt"))

        for f in tqdm(data_files, desc=f"Reading data in {d.name}", leave=False):
            station_id = int(f.name.split("_")[2])
            if station_id in metro_ids:
                logger.info(f"Reading data for station {station_id} from {f.name}")
                df = pd.read_csv(f, sep="\t", engine="python", skiprows=1)
                dfs.append(df)

    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        logger.info(f"Loaded {len(combined)} observational rows")
        return combined

    logger.warning("No observational data loaded")
    return pd.DataFrame()


# ------------------------------------------------------------
# 5. SAVE OUTPUTS
# ------------------------------------------------------------

def save_region_outputs(out_dir: Path, stations, data):
    """
    Save filtered station metadata and observational data.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving station metadata → {out_dir / 'stations.csv'}")
    stations.to_csv(out_dir / "stations.csv", index=False)

    logger.info(f"Saving observational data → {out_dir / 'observations.csv'}")
    data.to_csv(out_dir / "observations.csv", index=False)


# ------------------------------------------------------------
# 6. END-TO-END REGION PROCESSOR
# ------------------------------------------------------------

def process_region(region_path: Path, bbox, output_path: Path):
    """
    Full pipeline:
    - extract ZIPs
    - load station metadata
    - filter to metro stations
    - load observational data
    - save outputs
    """
    logger.info(f"Processing region: {region_path}")

    extracted = extract_all_zips(region_path)
    stations = load_station_metadata(extracted)
    metro = filter_metro(stations, bbox)
    metro_ids = metro["site_id"].astype(int).unique()
    data = load_metro_data(extracted, metro_ids)

    save_region_outputs(output_path, metro, data)

    logger.info(f"Finished processing region: {region_path}")

    return metro, data