__file_1__: "Handles the inputs and outputs"

import pandas as pd
import zipfile
import re
from pathlib import Path


import pandas as pd
import zipfile
import re
from pathlib import Path

STATION_ID_PATTERN = re.compile(r"(\d{6})")

def extract_station_id(path):
    name = Path(path).name
    match = STATION_ID_PATTERN.search(name)
    return match.group(1) if match else None

def _open_csv_from_zip(zip_path):
    with zipfile.ZipFile(zip_path) as z:
        csv_files = [f for f in z.namelist() if f.lower().endswith(".csv")]
        if len(csv_files) != 1:
            raise ValueError(f"Expected exactly one CSV in {zip_path}")
        with z.open(csv_files[0]) as f:
            yield f

def load_weather_chunks(path, chunksize=500_000):
    path = Path(path)
    if path.suffix.lower() == ".zip":
        for f in _open_csv_from_zip(path):
            for chunk in pd.read_csv(f, chunksize=chunksize):
                yield chunk
    else:
        for chunk in pd.read_csv(path, chunksize=chunksize):
            yield chunk