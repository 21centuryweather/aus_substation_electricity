#!/Users/mjl/anaconda/bin/python

__title__ = "Clean Melbourne and Brisbane 1/2 hourly electricity demand data and process with metadata"
__author__ = "Pia Vassallo"
__version__ = "2026_09_01"
__email__ = "pvas0009@student.monash.edu"

"""
Clean and load CSIRO NEAR substation electricity demand data
for Melbourne (VIC) and Brisbane (QLD).

This module provides:
- load_supplier_demand_and_metadata()
- load_state_demand_and_metadata()
- clean_data_sigma()
- clean_data_constant()
- linearly_fill_gaps()
- select_sites()

No printing, no logging, no script execution on import.
"""

from pathlib import Path
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------
# 1. LOAD A SINGLE SUPPLIER'S DEMAND + METADATA
# ---------------------------------------------------------------------

def load_supplier_demand_and_metadata(supplier_csv, metadata_df):
    """
    Load and clean a single supplier's CSIRO NEAR demand file,
    extract substation IDs, filter metadata to those IDs,
    and return cleaned demand + metadata.

    Args:
        supplier_csv (str or Path): path to one supplier's CSV file
        metadata_df (DataFrame): full DNSP metadata table (already filtered to state)

    Returns:
        demand_df (DataFrame): cleaned demand (wide, indexed by datetime)
        meta_subset (DataFrame): metadata for substations in this supplier
    """

    df = pd.read_csv(supplier_csv)

    # Build datetime index
    df["StartDeliveryTime"] = pd.to_datetime(df["StartDeliveryTime"], errors="coerce")
    df = df.dropna(subset=["StartDeliveryTime"])
    df = df.set_index("StartDeliveryTime")

    # Drop metadata columns
    df = df.drop(columns=["EndDeliveryTime", "UtilityName"], errors="ignore")

    # Extract NEAR-style IDs from column names
    # "'ABY'|'Amberley'|ActivePower" → "ABY"
    raw_cols = df.columns
    ids = [col.split("'")[1] for col in raw_cols]

    # Ensure IDs are unique
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate substation IDs in {supplier_csv}")

    # Rename columns
    df.columns = ids

    # Metadata uses "Zone Substation ID"
    if "Zone Substation ID" in metadata_df.columns:
        metadata_df = metadata_df.rename(columns={"Zone Substation ID": "ID"})

    meta_subset = metadata_df[metadata_df["ID"].isin(ids)].copy()

    return df, meta_subset


# ---------------------------------------------------------------------
# 2. LOAD ALL SUPPLIERS FOR A STATE
# ---------------------------------------------------------------------

def load_state_demand_and_metadata(state, state_dir, metadata_path):
    """
    Load and clean ALL suppliers for a state (QLD or VIC),
    merge each with metadata, and combine into one dataset.

    Args:
        state (str): "QLD" or "VIC"
        state_dir (str or Path): directory containing supplier folders
        metadata_path (str or Path): path to DNSP metadata CSV

    Returns:
        combined_demand (DataFrame): all suppliers combined (wide)
        combined_meta (DataFrame): metadata for all substations in the state
    """

    metadata_df = pd.read_csv(metadata_path)

    # Filter metadata to the state
    meta_state = metadata_df[metadata_df["State"] == state].copy()

    demand_list = []
    meta_list = []

    for supplier_folder in Path(state_dir).iterdir():
        if supplier_folder.is_dir():
            for csv_file in supplier_folder.glob("*.csv"):
                demand_df, meta_subset = load_supplier_demand_and_metadata(
                    csv_file, meta_state
                )
                demand_list.append(demand_df)
                meta_list.append(meta_subset)

    combined_demand = pd.concat(demand_list, axis=1)
    combined_meta = pd.concat(meta_list).drop_duplicates(subset="ID")

    return combined_demand, combined_meta


# ---------------------------------------------------------------------
# 3. CLEANING FUNCTIONS (OPTIONAL)
# ---------------------------------------------------------------------

def clean_data_sigma(df, sigma=5):
    """
    Remove values outside mean ± sigma * std.
    """
    mean = df.mean()
    std = df.std()
    lower = mean - sigma * std
    upper = mean + sigma * std
    return df.where((df > lower) & (df < upper))


def clean_data_constant(df, window="2h"):
    """
    Remove values where rolling std is extremely small (flatlined meters).
    """
    std = df.rolling(window=window).std()
    mean = df.mean()
    return df.where(std > mean / 1000)


def linearly_fill_gaps(ser_to_fill: pd.Series, max_gap=4) -> pd.Series:
    """
    Linearly fill gaps where gap length <= max_gap.
    """
    new_group_list = []
    ser_test = ser_to_fill.copy()

    if max_gap < len(ser_test):
        isna = pd.Series(np.where(ser_test.isna(), 1, np.nan), index=ser_test.index)
        isna_sum = isna.copy()
        for n in range(1, max_gap + 1):
            isna_sum = isna_sum + isna.shift(n)
        break_idxs = isna_sum.dropna().index

        prev_break = ser_test.index[0]

        for next_break in break_idxs:
            group = ser_test[prev_break:next_break]
            if group.count() == 0:
                continue
            new_group = group.interpolate(method="linear", limit=max_gap, limit_area="inside")
            new_group_list.append(new_group)
            prev_break = next_break

        group = ser_test[prev_break:]
    else:
        group = ser_test

    new_group = group.interpolate(method="linear", limit=max_gap, limit_area="inside")
    new_group_list.append(new_group)

    filled = pd.concat(new_group_list).sort_index()
    filled = filled[~filled.index.duplicated(keep="first")]

    assert len(filled) == len(ser_to_fill)
    return filled


# ---------------------------------------------------------------------
# 4. LAND-USE FILTERING (OPTIONAL)
# ---------------------------------------------------------------------

def select_sites(info, area_min=0, res_min=0, res_max=1, com_min=0, com_max=1,
                 ind_min=0, ind_max=1, farm_max=1):
    """
    Select substations based on area and land-use fractions.

    Args:
        info (DataFrame): metadata indexed by ID
    """

    # Ensure NEAR column name is correct
    if "Zone Substation Area (km²)" in info.columns:
        area_col = "Zone Substation Area (km²)"
    else:
        area_col = "Area"  # fallback

    filtered = info[
        (info[area_col] > area_min) &
        (info["Residential"] > res_min) &
        (info["Residential"] < res_max) &
        (info["Commercial"] > com_min) &
        (info["Commercial"] < com_max) &
        (info["Industrial"] > ind_min) &
        (info["Industrial"] < ind_max) &
        (info["Primary Production"] < farm_max)
    ]

    return filtered.index.to_list()

# ---------------------------------------------------------------------
# 5. SAVE CLEANED STATE DATA
# ---------------------------------------------------------------------

def save_cleaned_state(
    state,
    state_dir,
    metadata_path,
    output_root,
    sigma=None,
    remove_constant=False,
    fill_small_gaps=False,
    max_gap=4,
    landuse_filters=None
):
    """
    Load, clean, optionally filter, and save cleaned demand + metadata
    for a given state (QLD or VIC).

    Args:
        state (str): "QLD" or "VIC"
        state_dir (Path): directory containing raw supplier folders
        metadata_path (Path): path to DNSP metadata CSV
        output_root (Path): root folder for cleaned outputs
        sigma (float or None): sigma threshold for outlier removal
        remove_constant (bool): remove flatlined meters
        fill_small_gaps (bool): fill small gaps linearly
        max_gap (int): maximum gap length to fill
        landuse_filters (dict or None): optional land-use filtering
    """

    # 1. Load raw state data
    demand, meta = load_state_demand_and_metadata(state, state_dir, metadata_path)

    # 2. Optional cleaning
    if sigma is not None:
        demand = clean_data_sigma(demand, sigma=sigma)

    if remove_constant:
        demand = clean_data_constant(demand)

    if fill_small_gaps:
        demand = demand.apply(linearly_fill_gaps, max_gap=max_gap)

    # 3. Optional land-use filtering
    if landuse_filters is not None:
        selected_ids = select_sites(meta, **landuse_filters)
        demand = demand[selected_ids]
        meta = meta.loc[selected_ids]

    # 4. Prepare output directory
    out_dir = Path(output_root) / state
    out_dir.mkdir(parents=True, exist_ok=True)

    # 5. Save outputs
    demand.to_csv(out_dir / "demand.csv")
    meta.to_csv(out_dir / "metadata.csv")    