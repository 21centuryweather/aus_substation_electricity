#!/Users/mjl/anaconda/bin/python

__title__ = "Process and clean substation 1/2 hourly electricity data"
__author__ = "Mathew Lipson"
__version__ = "2024-10-23"
__email__ = "m.lipson@unsw.edu.au"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import glob
from pathlib import Path


pd.set_option('display.width', 150)

oshome=os.getenv('HOME')
projpath = f'.'
datapath = f'{projpath}/data/raw_data/VIC_demand/AustNet_VIC'
obspath =  f'{projpath}/data/cleaned_data/VIC/cleaned_obs'
plotpath = f'{projpath}/figures'

# create plotpath if it doesn't exist
if not os.path.exists(plotpath):
    os.makedirs(plotpath)

# year start and end date if wanted
sdate,edate = None, None

# select state domains
# domains = ['vic','nsw','qld','wa','tas','sa', 'act']
domains = ['vic']

#linearly fill gaps max size
fill_gaps = False

def load_single_station_obs(station_file):
    df = pd.read_parquet(station_file)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df

def map_substations_to_weather(sub_info, obs_path):
    # extract station metadata from filenames
    files = list(Path(obs_path).glob("*.parquet"))
    stations = []
    for f in files:
        station_id = f.name.split("_")[0]
        stations.append({"station_id": station_id, "file": f})

    stations = pd.DataFrame(stations)

    # you already have substation lat/lon in info
    # you need weather station lat/lon too (I can help you build this)

    # for now, assign ALL substations to ONE station to avoid crashes
    stations = stations.iloc[0]  # pick first station
    sub_info["nearest_station_file"] = stations["file"]

    return sub_infobha

def get_demand_data(suppliers, domain, obs_path, sdate, edate):
    print(f'processing {domain} substations for {suppliers} from {sdate} to {edate}')

    demand_list = []
    info_list = []

    for supplier in suppliers:
        print(supplier)

        # load demand + metadata
        demand, info = get_supplier_demand(supplier, domain=domain)

        # VIC: map substations to weather stations + load obs
        if domain == 'vic':
            info = map_substations_to_weather(info, obs_path)
            station_file = info["nearest_station_file"].iloc[0]
            obs = load_single_station_obs(station_file)
        else:
            obs = read_bom_half_hourly(obs_path)

        # clean demand using obs
        demand = clean_data(demand, obs)

        demand_list.append(demand)
        info_list.append(info)

    # combine info
    info_all = pd.concat(info_list, axis=0)
    info_all = info_all.sort_values(by='Residential', ascending=False)

    # combine demand
    demand_all = pd.concat(demand_list, axis=1)
    demand_all = demand_all[info_all.index.to_list()]
    demand_all = demand_all[sdate:edate]

    return demand_all, info_all, obs


def read_vic_cleaned_obs(obspath):
    """
    Reads cleaned VIC BoM obs from parquet files.
    Returns a single obs dataframe indexed by datetime.
    """

    files = sorted(Path(obspath).glob("*.parquet"))

    dfs = []
    for f in files:
        df = pd.read_parquet(f)

        # ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        dfs.append(df)

    obs = pd.concat(dfs).sort_index()
    return obs

def tidy_aws_columns(df):

    col_map = {
       'Latitude to four decimal places - in degrees'   : 'latitude',
       'Longitude to four decimal places - in degrees'  : 'longitude',
       'Year Month Day Hour Minutes in YYYY'            : 'year',
       'MM'                                             : 'month',
       'DD'                                             : 'day',
       'HH24'                                           : 'hour',
       'MI format in Local standard time'               : 'minute',
       'Air Temperature in degrees Celsius'             : 't2m',
       'Highest air temperature in last 30 minutes in degrees Celsius where observations count >= 12' : 't2m_30max',
       'Lowest air temperature in last 30 minutes in degrees Celsius where observations count >= 12'  : 't2m_30min',
    }

    df.columns = df.columns.str.strip()
    df = df.rename(columns=col_map)

    return df

def get_substation_data_vic(filename):
    df = pd.read_csv(filename)

    # If the file has only 1 column, it's a broken AusNet file
    if df.shape[1] == 1:
        print(f"Skipping malformed file: {filename}")
        return None, None

    # Identify the timestamp column
    possible_time_cols = [
        'StartDeliveryTime', 'START_DELIVERY_TIME',
        'start_time', 'StartTime', 'DateTime',
        'IntervalStart', 'START', 'Time'
    ]

    time_col = None
    for col in df.columns:
        if col in possible_time_cols:
            time_col = col
            break

    if time_col is None:
        raise ValueError(
            f"No valid timestamp column found in {filename}. "
            f"Columns were: {df.columns.tolist()}"
        )

    df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col)

    df = df.drop(columns=['EndDeliveryTime', 'UtilityName'], errors='ignore')

    clean_cols = []
    for col in df.columns:
        parts = col.split("\\")
        code = parts[0].strip("'").strip('"')
        clean_cols.append(code)

    df.columns = clean_cols

    info = pd.DataFrame({
        "ID": clean_cols,
        "Residential": 1,
    }).set_index("ID")

    return df, info


def get_domain_info(domain, projpath, datapath):

    if domain == 'vic':
        suppliers = ['AusNet_VIC']   # ONLY ONE
        supplier = 'AusNet_VIC'
        obs_path = f'{projpath}/data/cleaned_data/VIC/cleaned_obs'
        return suppliers, supplier, obs_path

    if domain == 'nsw':
        suppliers = ['ausgrid']
        supplier = 'ausgrid'
        obs_path = f'{datapath}/BOMdata/HD01D_Data_066194_541079810413811.txt'
        return suppliers, supplier, obs_path

    if domain == 'qld':
        suppliers = ['energex']
        supplier = 'energex'
        obs_path = f'{datapath}/BOMdata/HD01D_Data_009225_546889910504794.txt'
        return suppliers, supplier, obs_path

    if domain == 'wa':
        suppliers = ['western']
        supplier = 'western'
        obs_path = f'{datapath}/BOMdata/HD01D_Data_009225_546889910504794.txt'
        return suppliers, supplier, obs_path

    raise ValueError(f"Unknown domain: {domain}")

def get_supplier_info(supplier):
    """
    Load DNSP metadata for a given supplier.
    The DNSP file IS the metadata.
    Handles VIC naming mismatches (e.g., 'AusNet' vs 'AusNet Services').
    """

    # Load the DNSP metadata file
    metadata_path = f"{projpath}/data/DNSP_Zone_Substation_Characteristics.csv"
    metadata = pd.read_csv(metadata_path)

    # ---------------------------------------------------------
    # 1. Normalize DNSP names in metadata
    # ---------------------------------------------------------
    metadata["clean_dns_name"] = (
        metadata["Distribution Network Service Provider"]
        .str.lower()
        .str.replace("services", "", regex=False)
        .str.replace("service", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.strip()
    )

    # ---------------------------------------------------------
    # 2. Normalize the supplier name passed into the function
    # ---------------------------------------------------------
    supplier_clean = (
        supplier.lower()
        .replace("_vic", "")
        .replace("_", "")
        .replace("services", "")
        .replace("service", "")
        .strip()
    )

    # ---------------------------------------------------------
    # 3. Filter metadata for this supplier
    # ---------------------------------------------------------
    info = metadata[metadata["clean_dns_name"] == supplier_clean].copy()

    if info.empty:
        print(f"WARNING: No metadata found for supplier '{supplier}' "
              f"(normalized as '{supplier_clean}').")
        print("Available DNSP names in metadata:",
              metadata['clean_dns_name'].unique())
        return pd.DataFrame()

    # ---------------------------------------------------------
    # 4. Detect the ID column automatically
    # ---------------------------------------------------------
    possible_id_cols = [
        "ID", "Id", "id",
        "Substation ID", "SubstationID",
        "Zone Substation ID", "Zone_Substation_ID",
        "StationID", "Station ID",
        "Code", "Substation", "Zone Substation"
    ]

    id_col = None
    for col in info.columns:
        if col in possible_id_cols:
            id_col = col
            break

    if id_col is None:
        raise ValueError(
            f"No ID-like column found in DNSP metadata. "
            f"Columns were: {info.columns.tolist()}"
        )

    # ---------------------------------------------------------
    # 5. Set index to the detected ID column
    # ---------------------------------------------------------
    info = info.set_index(id_col)

    return info


def get_substation_data_vic(filename):
    # Load the CSV
    df = pd.read_csv(filename)

    # ---------------------------------------------------------
    # 1. Skip malformed AusNet files (e.g., 2017)
    # ---------------------------------------------------------
    if df.shape[1] == 1:
        print(f"Skipping malformed VIC file (only 1 column): {filename}")
        return None, None

    # ---------------------------------------------------------
    # 2. Identify the timestamp column (AusNet is inconsistent)
    # ---------------------------------------------------------
    possible_time_cols = [
        'StartDeliveryTime', 'START_DELIVERY_TIME',
        'start_time', 'StartTime', 'DateTime',
        'IntervalStart', 'START', 'Time'
    ]

    time_col = None
    for col in df.columns:
        if col in possible_time_cols:
            time_col = col
            break

    if time_col is None:
        raise ValueError(
            f"No valid timestamp column found in {filename}. "
            f"Columns were: {df.columns.tolist()}"
        )

    # ---------------------------------------------------------
    # 3. Parse timestamps (Australian format → dayfirst=True)
    # ---------------------------------------------------------
    df[time_col] = pd.to_datetime(
        df[time_col],
        dayfirst=True,
        errors='coerce'
    )

    # Drop rows where timestamp failed to parse
    df = df.dropna(subset=[time_col])

    # Set index
    df = df.set_index(time_col)

    # ---------------------------------------------------------
    # 4. Drop non‑demand columns if present
    # ---------------------------------------------------------
    df = df.drop(columns=['EndDeliveryTime', 'UtilityName'], errors='ignore')

    # ---------------------------------------------------------
    # 5. Clean AusNet column names
    #    "'BDL'\\Bairnsdale Zone Substation\\ActivePower"
    #    → "BDL"
    # ---------------------------------------------------------
    clean_cols = []
    for col in df.columns:
        parts = col.split("\\")
        code = parts[0].strip("'").strip('"')
        clean_cols.append(code)

    df.columns = clean_cols

    # ---------------------------------------------------------
    # 6. Build metadata table
    # ---------------------------------------------------------
    info = pd.DataFrame({
        "ID": clean_cols,
        "Residential": 1,   # placeholder until real metadata added
    }).set_index("ID")

    return df, info


def get_supplier_demand(
        supplier,
        area_min=0, res_min=0, res_max=1,
        com_min=0, com_max=1,
        ind_min=0, ind_max=1,
        farm_max=1,
        domain=None
    ):
    """
    Load demand + metadata for a supplier.
    Handles VIC (AusNet, Jemena, CitiPower, etc.) and NSW/QLD/WA formats.
    """

    # Load metadata (works for all states)
    info = get_supplier_info(supplier)

    # ---------------------------------------------------------
    # VIC BRANCH — uses raw_data/VIC_demand and VIC-specific loader
    # ---------------------------------------------------------
    if domain == 'vic':
        fnames = sorted(glob.glob(
            f'{datapath}/raw_data/VIC_demand/{supplier}/*.csv'
        ))

        data_info = []
        for fname in fnames:
            df, meta = get_substation_data_vic(fname)

            # Skip malformed AusNet files (e.g., 2017)
            if df is None:
                print(f"Skipping malformed file: {fname}")
                continue

            data_info.append((df, meta))

        if len(data_info) == 0:
            raise ValueError(f"No valid VIC files found for supplier {supplier}")

    # ---------------------------------------------------------
    # NSW / QLD / WA BRANCH — original behaviour
    # ---------------------------------------------------------
    else:
        fnames = sorted(glob.glob(
            f'{datapath}/{supplier}/collated_standardized_{supplier}*.csv'
        ))

        data_info = [(get_substation_data(fname)) for fname in fnames]

    # ---------------------------------------------------------
    # Combine demand + metadata
    # ---------------------------------------------------------
    demand = pd.concat([d for d, i in data_info], axis=1)
    info_from_files = pd.concat([i for d, i in data_info], axis=0)

    # ---------------------------------------------------------
    # Remove columns not in metadata
    # ---------------------------------------------------------
    print('following columns in demand are not in info index:')
    print(demand.columns[~demand.columns.isin(info.index)].tolist())
    print('removing these columns from demand')

    demand = demand.loc[:, demand.columns.isin(info.index)]

    print(f'number of substations in {supplier} substation info: {len(info)}')
    print(f'number of substations in {supplier} substation data: {len(demand.columns)}')

    # ---------------------------------------------------------
    # Select sites based on area + land use
    # ---------------------------------------------------------
    sites = info.loc[
        select_sites(
            info,
            area_min, res_min, res_max,
            com_min, com_max,
            ind_min, ind_max,
            farm_max
        )
    ].sort_values(by='Residential', ascending=False)

    print('following sites match selection criteria:')
    print(sites)

    # Filter demand + info to selected sites
    demand = demand.loc[:, demand.columns.isin(sites.index)]
    info = info.loc[demand.columns]

    return demand, info

def clean_data(demand_orig,obs):

    demand = demand_orig.copy()

    # create temperature bins for later grouping
    bins = [-np.inf] + list(range(0, 55, 5)) + [np.inf]
    labels,key = ['<0'] + [f'{i}-{i+4}' for i in range(0, 50, 5)] + ['>50'], 't2m'
    obs[f'{key}_bin'] = pd.cut(obs[key], bins=bins, labels=labels)

    print('removing negative values')
    demand = demand.where(demand>0)

    print(f'removing values outside of 5 standard deviations, within {key} bins')
    demand = demand.groupby(obs[f'{key}_bin'], observed=False, group_keys=False).apply(clean_data_sigma,sigma=5)

    print('removing constant values')
    demand = clean_data_constant(demand,window='2h')

    if fill_gaps:
        print('linearly filling gaps')
        demand = demand.apply(linearly_fill_gaps, max_gap=4, result_type='expand')
    else:
        print('not filling gaps, set fill_gaps=True to enable')

    return demand

def clean_data_sigma(df,sigma):
    '''a function that cleans data outside of x standard deviations'''
    # calculate mean and standard deviation
    mean = df.mean()
    std = df.std()
    # calculate upper and lower bounds
    lower = mean - sigma*std
    upper = mean + sigma*std
    # replace values outside of bounds with nan
    df = df.where((df > lower) & (df < upper))

    return df

def clean_data_constant(df,window='2h'):
    '''a function that cleans data that is constant for more than x hours'''
    # calculate rolling standard deviation
    std = df.rolling(window=window).std()
    # replace values outside of bounds with nan
    mean = df.mean()
    df = df.where(std > mean/1000)

    return df

def linearly_fill_gaps(ser_to_fill : pd.Series, max_gap=4) -> pd.Series:
    ''' linearly fill gaps where gap is smaller than max_gap
    args
        ser_to_fill (pd.Series): the series to fill
        max_gap (int): the maximum gap to fill
    return
        filled (pd.Series): the filled series
    '''

    new_group_list = []

    ser_test = ser_to_fill.copy()

    # break series into groups (unless series is shorter than max_gap)
    if max_gap < len(ser_test):

        # find break points
        isna = pd.Series( np.where(ser_test.isna(), 1, np.nan), index=ser_test.index )
        isna_sum = isna
        for n in range(1,max_gap+1):
            isna_sum = isna_sum + isna.shift(n)
        break_idxs = isna_sum.dropna().index

        # # add start series
        prev_break = ser_test.index[0]
        
        for next_break in break_idxs:
            group = ser_test[prev_break:next_break]

            # skip to next loop if no values in group (for efficiency)
            if group.count() == 0:
                continue

            new_group = group.interpolate(method='linear',limit=max_gap, limit_area='inside')
            new_group_list.append(new_group)

            prev_break = next_break

        # append final group without interpolation
        group = ser_test[prev_break:]

    else: #simply group entire series
        group = ser_test

    new_group = group.interpolate(method='linear',limit=max_gap, limit_area='inside')
    new_group_list.append(new_group)

    # concatenate all groups
    filled = pd.concat(new_group_list).sort_index()
    filled = filled[~filled.index.duplicated(keep='first')]

    assert len(filled) == len(ser_to_fill), 'length of filled series is different to original'
    print('values filled linearly: %s ' %(filled.count() - ser_to_fill.count()))

    return filled

def select_sites(info,area_min,res_min,res_max,com_min,com_max,ind_min,ind_max,farm_max):
    '''
    Selects sites based on area and land use
    args:
        info (dataframe): the supplier info dataframe
        area_min (float): minimum area of site
        res_min (float): minimum residential fraction of site
        res_max (float): maximum residential fraction of site
        com_min (float): minimum commercial fraction of site
        com_max (float): maximum commercial fraction of site
        ind_min (float): minimum industrial fraction of site
        ind_max (float): maximum industrial fraction of site
        farm_max (float): maximum primary production fraction
    return:
        sites (list): list of sites
    '''

    # select sites based on area and land use
    sites = info[(info['Area']>area_min) & 
                (info['Residential']>res_min) & 
                (info['Residential']<res_max) &
                (info['Commercial']>com_min) &
                (info['Commercial']<com_max) & 
                (info['Industrial']>ind_min) &
                (info['Industrial']<ind_max) &
                (info['Primary Production']<farm_max)].index.to_list()

    return sites

###############################################################################

if __name__ == '__main__':

    projpath = '.'
    datapath = f'{projpath}/data'

    domain = 'vic'   # ONLY ONE DOMAIN
    sdate = None
    edate = None

    suppliers, supplier, obs_path = get_domain_info(domain, projpath, datapath)

    demand, info, obs = get_demand_data(suppliers, domain, obs_path, sdate, edate)

    print('\nsubstation info:\n', info)
