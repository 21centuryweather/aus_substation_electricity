import pandas as pd
import numpy as np
from pathlib import Path
import re
from tqdm import tqdm


# --------------------------------------------
# Load and clean one yearly CSV file
# --------------------------------------------

def load_single_year_file(csv_path, metadata_df):
    df = pd.read_csv(csv_path)

    # Identify timestamp column
    time_col = "StartDeliveryTime"
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col])
    df = df.drop_duplicates(subset=[time_col])

    # Set index
    df = df.set_index(time_col)

    #ensure monotonic index for rolling()
    df = df.sort_index()

    # Drop irrelevant columns
    df = df.drop(columns=["EndDeliveryTime", "UtilityName"], errors="ignore")

    # Clean column names
    df.columns = [col.split("|")[0].replace("'", "").strip() for col in df.columns]

    # Subset metadata
    meta_subset = metadata_df[metadata_df["ID"].isin(df.columns)].copy()

    return df, meta_subset


# --------------------------------------------
# Cleaning functions
# --------------------------------------------

def clean_data_sigma(df, sigma=5):
    mean = df.mean()
    std = df.std()
    lower = mean - sigma * std
    upper = mean + sigma * std

    cleaned = pd.DataFrame(index=df.index)
    for col in tqdm(df.columns, desc="Sigma cleaning", unit="col"):
        cleaned[col] = df[col].where((df[col] > lower[col]) & (df[col] < upper[col]))
    return cleaned


def clean_data_constant(df, window="2h"):
    cleaned = pd.DataFrame(index=df.index)
    mean = df.mean()

    for col in tqdm(df.columns, desc="Constant-value cleaning", unit="col"):
        std = df[col].rolling(window=window).std()
        cleaned[col] = df[col].where(std > mean[col] / 1000)
    return cleaned


def linearly_fill_gaps(ser, max_gap=4):
    filled = ser.interpolate(method="linear", limit=max_gap, limit_area="inside")
    return filled


def fill_gaps_df(df, max_gap=4):
    filled = pd.DataFrame(index=df.index)
    for col in tqdm(df.columns, desc="Gap filling", unit="col"):
        filled[col] = linearly_fill_gaps(df[col], max_gap=max_gap)
    return filled


# --------------------------------------------
# Process all yearly files for one DNSP
# --------------------------------------------

def process_DNSP_years(
    dnspsubfolder,
    state_dir,
    metadata_path,
    output_root,
    sigma=5,
    remove_constant=True,
    fill_small_gaps=True,
    max_gap=4
):
    import re

    print(f"\n--- Starting DNSP: {dnspsubfolder} ---")

    # Setup paths
    input_dir = Path(state_dir) / dnspsubfolder
    output_dir = Path(output_root) / dnspsubfolder
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    metadata_df = pd.read_csv(metadata_path)

    # Standardize ID column
    if "Zone Substation ID" in metadata_df.columns:
        metadata_df = metadata_df.rename(columns={"Zone Substation ID": "ID"})

    # -------------------------------
    # Metadata matching (robust)
    # -------------------------------
    raw_name = dnspsubfolder.split("_")[0]

    # Convert camelCase → spaced (UnitedEnergy → United Energy)
    raw_name_spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", raw_name)

    # Fuzzy match
    mask = metadata_df["Distribution Network Service Provider"].str.contains(
        raw_name_spaced, case=False, na=False
    )

    matched_meta = metadata_df[mask].copy()

    # Save metadata
    meta_out = output_dir / f"{dnspsubfolder}_metadata.csv"
    matched_meta.to_csv(meta_out, index=False)

    print(f"  Matched {len(matched_meta)} metadata rows for '{raw_name_spaced}'")
    if len(matched_meta) == 0:
        print("  WARNING: No metadata matched — check naming")

    # -------------------------------
    # Process each CSV file
    # -------------------------------
    csv_files = sorted(input_dir.glob("*.csv"))

    for csv_file in csv_files:
        print(f"\nProcessing file: {csv_file.name}")

        # Extract year
        match = re.search(r"(20\d{2})", csv_file.name)
        if not match:
            print(f"  Skipping {csv_file.name} — no year found")
            continue
        year = match.group(1)

        # Load and clean raw file
        df, meta_subset = load_single_year_file(csv_file, metadata_df)

        # Cleaning steps
        if sigma is not None:
            df = clean_data_sigma(df, sigma=sigma)

        if remove_constant:
            df = clean_data_constant(df)

        if fill_small_gaps:
            df = fill_gaps_df(df, max_gap=max_gap)

        # Save cleaned output
        out_path = output_dir / f"{dnspsubfolder}_{year}_cleaned.csv"
        df.to_csv(out_path)

        print(f"  Saved cleaned file: {out_path.name}")


# --------------------------------------------
# Run all VIC DNSPs
# --------------------------------------------

def process_all_VIC():
    state_dir = "/home/565/pv3484/aus_substation_electricity/data/raw_data/VIC_demand"
    metadata_path = "/home/565/pv3484/aus_substation_electricity/data/DNSP_Zone_Substation_Characteristics.csv"
    output_root = "/home/565/pv3484/aus_substation_electricity/data/cleaned_data/VIC"

    dnsps = ["CitiPower_VIC", "AusNet_VIC", "Jemena_VIC", "Powercor_VIC", "UnitedEnergy_VIC"]

    for dnspsubfolder in dnsps:
        process_DNSP_years(
            dnspsubfolder=dnspsubfolder,
            state_dir=state_dir,
            metadata_path=metadata_path,
            output_root=output_root,
            sigma=5,
            remove_constant=True,
            fill_small_gaps=True,
            max_gap=4
        )