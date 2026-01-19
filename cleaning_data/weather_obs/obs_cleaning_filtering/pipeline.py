__file_3: "Puts all the work together and meshes it with metadata"

"""
Puts all the work together and meshes it with metadata.
Handles:
- station ID extraction
- metadata lookup
- chunked loading → cleaning → filtering
- safe Parquet writing with PyArrow
"""

from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

from input_output import load_weather_chunks, extract_station_id
from cleaning import clean_chunk
from filtering import filter_by_year
from metadata import load_station_metadata, resolve_station_metadata


def process_file(path, output_dir, metadata_df):
    """
    Process a single raw weather file:
        - detect station ID
        - validate against metadata
        - stream chunks → clean → filter
        - append to a station-level Parquet file (correctly)
    """

    path = Path(path)
    station_id = extract_station_id(path)

    if station_id is None:
        print(f"Skipping {path.name}: no station ID found")
        return

    # Metadata lookup
    meta = resolve_station_metadata(station_id, metadata_df)
    if meta is None:
        print(f"Skipping {path.name}: station ID {station_id} not in metadata")
        return

    # Correct metadata key
    station_name = meta.get("name", "unknown_station")

    # Output file path
    out_path = Path(output_dir) / f"{station_id}_{station_name}_cleaned.parquet"

    writer = None  # pyarrow writer

    for chunk in load_weather_chunks(path):
        # Clean + filter
        chunk = clean_chunk(chunk)
        chunk = filter_by_year(chunk)

        if chunk.empty:
            continue

        # Force all non-datetime columns to string to avoid ArrowTypeError
        chunk = chunk.astype({col: "string" for col in chunk.columns if col != "datetime"})

        # Convert to Arrow table
        table = pa.Table.from_pandas(chunk)

        # Create writer on first non-empty chunk
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema)

        writer.write_table(table)

    # Close writer if any data was written
    if writer is not None:
        writer.close()
        print(f"Finished {path.name} → {out_path.name}")
    else:
        print(f"No valid data for {path.name} (all chunks empty after filtering)")


def process_region(input_paths, output_dir, metadata_path):
    """
    Process all files for a region (e.g., QLD or VIC).
    Creates the output directory if needed.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_df = load_station_metadata(metadata_path)

    for path in input_paths:
        process_file(path, output_dir, metadata_df)