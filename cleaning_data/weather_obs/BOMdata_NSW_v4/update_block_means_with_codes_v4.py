import pandas as pd

# 1. Load small lookup tables

mapping = pd.read_csv(
    "/home/565/pv3484/aus_substation_electricity/pia_notebooks/NSW_data/"
    "relative_ranking/substation_to_weather_mapping.csv"
)

rank = pd.read_csv(
    "/home/565/pv3484/aus_substation_electricity/data/cleaned_data/"
    "full_nsw_relative_rank.csv"
)

mapping_small = mapping[["nearest_station_id", "name"]].copy()
rank_small = rank[["station_name", "station_code"]].copy()

mapping_small["nearest_station_id"] = (
    mapping_small["nearest_station_id"].astype(str).str.strip()
)
mapping_small["name"] = mapping_small["name"].str.strip()

rank_small["station_name"] = rank_small["station_name"].str.strip()
rank_small["station_code"] = rank_small["station_code"].astype(str).str.strip()

# Unique lookup
rank_lookup = rank_small.drop_duplicates(subset=["station_name"]).copy()

# 2. Output file (v4)

output_path = (
    "/g/data/ng72/pv3484/substation_data/BOM_NSW_weather_processed_v4/"
    "weather_holiday_block_means_with_codes_v4.csv"
)

header_written = False

# 3. Chunked merge

chunk_size = 100_000
chunk_idx = 0

for chunk in pd.read_csv(
    "/g/data/ng72/pv3484/substation_data/BOM_NSW_weather_processed_v4/"
    "weather_holiday_block_means_v4.csv",
    chunksize=chunk_size,
):
    chunk_idx += 1
    print(f"Processing chunk {chunk_idx}...")

    chunk["station_id"] = chunk["station_id"].astype(str).str.strip()

    # Merge 1: weather → mapping
    merged = chunk.merge(
        mapping_small,
        left_on="station_id",
        right_on="nearest_station_id",
        how="left",
        validate="many_to_many",
    )

    # Merge 2: mapping → rank_lookup
    merged = merged.merge(
        rank_lookup,
        left_on="name",
        right_on="station_name",
        how="left",
        validate="many_to_many",
    )

    merged.to_csv(
        output_path,
        mode="a",
        index=False,
        header=not header_written,
    )
    header_written = True
    print(f"Chunk {chunk_idx} written.")

print("Done.")
