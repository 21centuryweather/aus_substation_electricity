__file_3: "Puts all the work together and meshes it with metadata"

from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

from input_output import load_weather_chunks, extract_station_id
from cleaning import clean_chunk
from filtering import filter_by_year
from metadata import load_station_metadata, resolve_station_metadata


def process_file(path, output_dir, metadata_df):

    path = Path(path)
    station_id = extract_station_id(path)

    if station_id is None:
        print(f"Skipping {path.name}: no station ID found")
        return

    meta = resolve_station_metadata(station_id, metadata_df)
    if meta is None:
        print(f"WARNING: {path.name} → station ID {station_id} not found in ANY metadata source")
        return

    station_name = meta.get("name", "unknown_station")
    out_path = Path(output_dir) / f"{station_id}_{station_name}_cleaned.parquet"

    writer = None

    for chunk in load_weather_chunks(path):
        chunk = clean_chunk(chunk)
        chunk = filter_by_year(chunk)

        if chunk.empty:
            continue

        # Inject metadata (lat, lon, name, etc.)
        for key, value in meta.items():
            if key not in chunk.columns:
                chunk[key] = value

        chunk = chunk.astype({col: "string" for col in chunk.columns if col != "datetime"})

        table = pa.Table.from_pandas(chunk)

        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema)

        writer.write_table(table)

    if writer is not None:
        writer.close()
        print(f"Finished {path.name} → {out_path.name}")
    else:
        print(f"No valid data for {path.name}")


def process_region(input_paths, output_dir, metadata_path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_df = load_station_metadata(metadata_path)

    for path in input_paths:
        process_file(path, output_dir, metadata_df)