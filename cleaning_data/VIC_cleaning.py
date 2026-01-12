from pathlib import Path
import pandas as pd
import numpy as np


# ---------------------------------------------------------
# 1. Load a single VIC supplier file
# ---------------------------------------------------------

def load_supplier_demand_and_metadata_VIC(supplier_csv, metadata_df):
    """
    Load one VIC DNSP CSV and align it with metadata.
    """

    df = pd.read_csv(supplier_csv)

    # Identify timestamp column
    possible_time_cols = ["StartDeliveryTime", "DateTime", "IntervalStart"]
    time_col = next((c for c in possible_time_cols if c in df.columns), None)
    if time_col is None:
        raise KeyError(f"No usable timestamp column in {supplier_csv}")

    # Parse timestamps
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col])
    df = df.drop_duplicates(subset=[time_col])
    df = df.set_index(time_col)

    # Drop irrelevant columns
    df = df.drop(columns=["EndDeliveryTime", "UtilityName"], errors="ignore")

    # Clean column names (remove trailing metadata)
    clean_cols = [col.split("|")[0].replace("'", "") for col in df.columns]
    df.columns = clean_cols

    # Ensure metadata uses "ID"
    if "Zone Substation ID" in metadata_df.columns:
        metadata_df = metadata_df.rename(columns={"Zone Substation ID": "ID"})

    # Subset metadata to only IDs present in this file
    meta_subset = metadata_df[metadata_df["ID"].isin(clean_cols)].copy()

    return df, meta_subset


# ---------------------------------------------------------
# 2. Load all VIC DNSPs
# ---------------------------------------------------------

def load_state_demand_and_metadata_VIC(state_dir, metadata_path):
    """
    Load all VIC DNSP CSVs and combine into one demand + metadata set.
    """

    metadata_df = pd.read_csv(metadata_path)

    # VIC DNSPs
    dnsps = ["AusNet", "CitiPower", "Powercor", "Jemena", "United Energy"]

    # Filter metadata to VIC only
    meta_state = metadata_df[
        metadata_df["Distribution Network Service Provider"].isin(dnsps)
    ].copy()

    demand_list = []
    meta_list = []

    for supplier_folder in Path(state_dir).iterdir():
        if not supplier_folder.is_dir():
            continue

        # Recursively find all CSVs
        for csv_file in supplier_folder.rglob("*.csv"):
            demand_df, meta_subset = load_supplier_demand_and_metadata_VIC(
                csv_file, meta_state
            )
            demand_list.append(demand_df)
            meta_list.append(meta_subset)

    combined_demand = pd.concat(demand_list, axis=1)
    combined_meta = pd.concat(meta_list).drop_duplicates(subset="ID")

    return combined_demand, combined_meta


# ---------------------------------------------------------
# 3. Cleaning functions
# ---------------------------------------------------------

def clean_data_sigma(df, sigma=5):
    """Remove values outside ±sigma standard deviations."""
    mean = df.mean()
    std = df.std()
    lower = mean - sigma * std
    upper = mean + sigma * std
    return df.where((df > lower) & (df < upper))


def clean_data_constant(df, window="2h"):
    """Remove values that are constant over a rolling window."""
    std = df.rolling(window=window).std()
    mean = df.mean()
    return df.where(std > mean / 1000)


def linearly_fill_gaps(ser_to_fill: pd.Series, max_gap=4) -> pd.Series:
    """Fill short gaps linearly (max_gap half-hour steps)."""

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

    return filled


# ---------------------------------------------------------
# 4. Land-use site selection
# ---------------------------------------------------------

def select_sites(info,
                 area_min=0,
                 res_min=0, res_max=1,
                 com_min=0, com_max=1,
                 ind_min=0, ind_max=1,
                 farm_max=1):
    """Filter substations by land-use characteristics."""

    sites = info[
        (info["Area"] > area_min) &
        (info["Residential"] > res_min) &
        (info["Residential"] < res_max) &
        (info["Commercial"] > com_min) &
        (info["Commercial"] < com_max) &
        (info["Industrial"] > ind_min) &
        (info["Industrial"] < ind_max) &
        (info["Primary Production"] < farm_max)
    ].index.to_list()

    return sites


# ---------------------------------------------------------
# 5. Main VIC processing function
# ---------------------------------------------------------

def process_VIC_demand(
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
    Load, clean, filter, and save VIC substation demand + metadata.
    Returns:
        demand (DataFrame)
        info (DataFrame)
    """

    # Load raw VIC data
    demand, info = load_state_demand_and_metadata_VIC(state_dir, metadata_path)

    # Cleaning pipeline
    if sigma is not None:
        demand = clean_data_sigma(demand, sigma=sigma)

    if remove_constant:
        demand = clean_data_constant(demand)

    if fill_small_gaps:
        demand = demand.apply(linearly_fill_gaps, max_gap=max_gap)

    # Land-use filtering
    if landuse_filters is not None:
        selected_ids = select_sites(info, **landuse_filters)
        demand = demand.loc[:, demand.columns.isin(selected_ids)]
        info = info.loc[selected_ids]

    # Save cleaned outputs
    out_dir = Path(output_root) / "VIC"
    out_dir.mkdir(parents=True, exist_ok=True)

    demand.to_csv(out_dir / "demand.csv")
    info.to_csv(out_dir / "metadata.csv")

    return demand, info