"""
weather_processing.py

Memory-safe processing of BOM HD01D station metadata and observational data.

- Extracts all ZIPs
- Streams ALL station metadata to a single CSV
- Streams ALL observational data to a single CSV
- No bounding-box filtering
- No in-memory concatenation of all observations
"""

from pathlib import Path
import pandas as pd
import zipfile
from tqdm import tqdm


# ------------------------------------------------------------
# 1. ZIP extraction
# ------------------------------------------------------------

def extract_zip(zip_path: Path) -> Path:
    """
    Extract a single ZIP file into a folder with the same name (no .zip).
    Returns the extraction directory.
    """
    extract_dir = zip_path.with_suffix("")
    extract_dir.mkdir(exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)

    return extract_dir


def extract_all_zips(raw_dir: Path) -> list[Path]:
    """
    Extract all ZIPs in a directory.
    Returns a list of extracted directories.
    """
    zip_files = sorted(raw_dir.glob("*.zip"))
    extracted_dirs = []

    for z in tqdm(zip_files, desc="Extracting ZIPs"):
        extracted_dirs.append(extract_zip(z))

    return extracted_dirs


# ------------------------------------------------------------
# 2. Station metadata (streamed to CSV)
# ------------------------------------------------------------

def process_all_station_metadata(extracted_dirs: list[Path], out_path: Path) -> Path:
    """
    Load all HD01D_StnDet files and stream them into a single CSV.
    Returns the output CSV path.
    """
    stn_files = []
    for d in extracted_dirs:
        stn_files.extend(d.glob("HD01D_StnDet_*.txt"))

    header_written = False

    for f in tqdm(stn_files, desc="Processing station metadata"):
        df = pd.read_csv(
            f,
            sep=",",
            header=None,
            skiprows=1,
            engine="python",
            encoding="utf-8"
        )

        n_cols = df.shape[1]
        base_cols = ["station_number", "name", "lat", "lon"]
        extra_cols = [f"extra_{i}" for i in range(n_cols - 4)]
        df.columns = base_cols + extra_cols

        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
        df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

        df.to_csv(out_path, mode="a", index=False, header=not header_written)
        header_written = True

    return out_path


# ------------------------------------------------------------
# 3. Observational data (streamed to CSV)
# ------------------------------------------------------------

def process_all_observations(extracted_dirs: list[Path], out_path: Path, chunksize: int = 200_000) -> Path:
    """
    Load all HD01D_Data files and stream them into a single CSV.
    Uses chunked reading to stay memory-safe.
    Returns the output CSV path.
    """
    data_files = []
    for d in extracted_dirs:
        data_files.extend(d.glob("HD01D_Data_*.txt"))

    header_written = False

    for f in tqdm(data_files, desc="Processing observational data"):
        # Chunked reading for memory safety
        for chunk in pd.read_csv(
            f,
            sep=",",
            header=None,
            skiprows=1,
            engine="python",
            encoding="utf-8",
            chunksize=chunksize
        ):
            n_cols = chunk.shape[1]
            base_cols = ["station_number", "date"]
            extra_cols = [f"var_{i}" for i in range(1, n_cols - 1)]
            chunk.columns = base_cols + extra_cols

            chunk.to_csv(out_path, mode="a", index=False, header=not header_written)
            header_written = True

    return out_path


# ------------------------------------------------------------
# 4. Region processing pipeline (no filtering, streaming)
# ------------------------------------------------------------

def process_region(raw_dir: Path, bbox: tuple, out_dir: Path):
    """
    Full pipeline WITHOUT ANY FILTERING, memory-safe:

    - extract ZIPs
    - stream ALL station metadata to stations.csv
    - stream ALL observations to observations.csv

    The bbox argument is ignored (kept only for API compatibility).

    Returns:
        stations_csv_path, observations_csv_path
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Extract ZIPs
    extracted_dirs = extract_all_zips(raw_dir)

    # 2. Stream station metadata
    stations_out = out_dir / "stations.csv"
    if stations_out.exists():
        stations_out.unlink()
    stations_csv = process_all_station_metadata(extracted_dirs, stations_out)

    # 3. Stream observations
    obs_out = out_dir / "observations.csv"
    if obs_out.exists():
        obs_out.unlink()
    obs_csv = process_all_observations(extracted_dirs, obs_out)

    return stations_csv, obs_csv