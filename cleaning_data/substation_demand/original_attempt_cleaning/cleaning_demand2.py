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
- save_cleaned_state()

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
    """

    # ---------------------------------------------------------
    # 0. Load CSV
    # ---------------------------------------------------------
    df = pd.read_csv(supplier_csv)

    # Drop known non‑demand columns in Ergon files
    df = df.drop(columns=["Unnamed: 2", "Usage"], errors="ignore")

    # ---------------------------------------------------------
    # 1. Identify timestamp column across DNSP formats
    # ---------------------------------------------------------
    possible_time_cols = [
        "StartDeliveryTime",  # Energex / VIC NEAR
        "Start",              # Ergon
        "t_start",
        "DateTime",
        "IntervalStart"
    ]

    time_col = None
    for col in possible_time_cols:
        if col in df.columns:
            time_col = col
            break

    if time_col is None:
        raise KeyError(f"No usable timestamp column found in {supplier_csv}")

    # Convert to datetime and set index
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col])
    df = df.set_index(time_col)

    # ---------------------------------------------------------
    # 2. Drop metadata columns if present (Energex/VIC)
    # ---------------------------------------------------------
    df = df.drop(columns=["EndDeliveryTime", "UtilityName"], errors="ignore")

    # ---------------------------------------------------------
    # 3. Extract substation IDs for both formats
    # ---------------------------------------------------------
    clean_cols = []
    for col in df.columns:
        if "|" in col:
            # NEAR Energex/VIC format: "'ABY'|'Amberley'|ActivePower"
            clean_cols.append(col.split("|")[0].replace("'", ""))
        else:
            # Ergon simple format: "Aitkenvale"
            clean_cols.append(col)

    df.columns = clean_cols

    # ---------------------------------------------------------
    # 4. Align metadata
    # ---------------------------------------------------------
    if "Zone Substation ID" in metadata_df.columns:
        metadata_df = metadata_df.rename(columns={"Zone Substation ID": "ID"})

    meta_subset = metadata_df[metadata_df["ID"].isin(clean_cols)].copy()

    return df, meta_subset


# ---------------------------------------------------------------------
# 2. LOAD ALL SUPPLIERS FOR A STATE
# ---------------------------------------------------------------------

def load_state_demand_and_metadata(state, state_dir, metadata_path):
    """
    Load and clean ALL suppliers for a state (QLD or VIC),
    merge each with metadata, and combine into one dataset.
    """

    # Load metadata
    metadata_df = pd.read_csv(metadata_path)

    # Filter metadata by DNSP (metadata has no State column)
    if state == "QLD":
        dnsps = ["Energex", "Ergon"]
    elif state == "VIC":
        dnsps = ["AusNet", "CitiPower", "Powercor", "Jemena", "United Energy"]
    else:
        raise ValueError(f"Unknown state: {state}")

    meta_state = metadata_df[
        metadata_df["Distribution Network Service Provider"].isin(dnsps)
    ].copy()

    demand_list = []
    meta_list = []

    # Loop through supplier folders
    for supplier_folder in Path(state_dir).iterdir():

        # Skip non-folders
        if not supplier_folder.is_dir():
            continue

        # Skip Energex AC-estimated folder entirely
        if "AC_QLD" in supplier_folder.name or "AC_filtered" in supplier_folder.name:
            continue

        # Loop through CSVs inside folder
        for csv_file in supplier_folder.glob("*.csv"):

            # Skip AC-estimated files inside any folder
            if "AC_filtered" in csv_file.name:
                continue

            # Load and clean this supplier's data
            demand_df, meta_subset = load_supplier_demand_and_metadata(
                csv_file, meta_state
            )

            demand_list.append(demand_df)
            meta_list.append(meta_subset)

    # Combine all suppliers
    combined_demand = pd.concat(demand_list, axis=1)
    combined_meta = pd.concat(meta_list).drop_duplicates(subset="ID")

    return combined_demand, combined_meta

# ---------------------------------------------------------------------
# 3. CLEANING FUNCTIONS (OPTIONAL)
# ---------------------------------------------------------------------

def clean_data_sigma(df, sigma=5):
    """Remove values outside mean ± sigma * std."""
    mean = df.mean()
    std = df.std()
    lower = mean - sigma * std
    upper = mean + sigma * std
    return df.where((df > lower) & (df < upper))


def clean_data_constant(df, window="2h"):
    """Remove values where rolling std is extremely small (flatlined meters)."""
    std = df.rolling(window=window).std()
    mean = df.mean()
    return df.where(std > mean / 1000)


def linearly_fill_gaps(ser_to_fill: pd.Series, max_gap=4) -> pd.Series:
    """Linearly fill gaps where gap length <= max_gap."""
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
    """Select substations based on area and land-use fractions."""

    if "Zone Substation Area (km²)" in info.columns:
        area_col = "Zone Substation Area (km²)"
    else:
        area_col = "Area"

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
# 5. SAVE CLEANED DATA
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
    """

    demand, meta = load_state_demand_and_metadata(state, state_dir, metadata_path)

    if sigma is not None:
        demand = clean_data_sigma(demand, sigma=sigma)

    if remove_constant:
        demand = clean_data_constant(demand)

    if fill_small_gaps:
        demand = demand.apply(linearly_fill_gaps, max_gap=max_gap)

    if landuse_filters is not None:
        selected_ids = select_sites(meta, **landuse_filters)
        demand = demand[selected_ids]
        meta = meta.loc[selected_ids]

    out_dir = Path(output_root) / state
    out_dir.mkdir(parents=True, exist_ok=True)

    demand.to_csv(out_dir / "demand.csv")
    meta.to_csv(out_dir / "metadata.csv")