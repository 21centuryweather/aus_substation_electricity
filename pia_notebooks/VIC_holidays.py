#__title__: "Holiday Dictionary for Victoria"
#__purpose__: "Create a dictionary of all National and Victorian public holidays so I can call upon it in notebooks rather than copying and pasting the library each time"

#__Victorian_hoiday_list:
# Labour Day is observed on the second Monday in March
# Melbourne Cup is observed on the First Tuesday in November
# AFL Grand Final (25/09/15, 30/09/16, 29/09/17)
# King's Birthday is observed on the second Monday in June

#--- Helper Function to calculate Labour Day, Melb Cup and King's Birthday ---

import pandas as pd
from datetime import date, timedelta
from dateutil.easter import easter

def nth_weekday_of_month(year, month, weekday, n):
    """
    Return the date of the nth weekday of a given month.
    weekday: Monday=0 ... Sunday=6
    n: 1=first, 2=second, etc.
    """
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return pd.Timestamp(first + timedelta(days=offset + 7*(n-1)))

def labour_day_vic(year):
    # Second Monday in March
    return nth_weekday_of_month(year, 3, weekday=0, n=2)

def kings_birthday_vic(year):
    # Second Monday in June
    return nth_weekday_of_month(year, 6, weekday=0, n=2)

def melbourne_cup_vic(year):
    # First Tuesday in November
    return nth_weekday_of_month(year, 11, weekday=1, n=1)

#--- AFL Grand Final Public Holiday only came in from 2015 ---

AFL_GF_EVE_DATES = {
    2015: pd.Timestamp("2015-10-02"),
    2016: pd.Timestamp("2016-09-30"),
    2017: pd.Timestamp("2017-09-29"),
}

def afl_grand_final_eve(year):
    """
    Return the AFL Grand Final Eve public holiday for years it existed.
    For all other years, return NaT.
    """
    return AFL_GF_EVE_DATES.get(year, pd.NaT)

#--- Helper Function: Substitue Monday as the public holiday if the date falls on the weekend ---
def substitute_if_weekend(ts):
    """
    If a holiday falls on Saturday or Sunday, return the following Monday.
    Otherwise return the original date.
    """
    if ts.weekday() == 5:  # Saturday
        return ts + pd.Timedelta(days=2)
    if ts.weekday() == 6:  # Sunday
        return ts + pd.Timedelta(days=1)
    return ts

#--- Christmas, Boxing Day and Easter substitute logic ---
def christmas_day_vic(year):
    ts = pd.Timestamp(f"{year}-12-25")
    return substitute_if_weekend(ts)

def boxing_day_vic(year):
    ts = pd.Timestamp(f"{year}-12-26")
    return substitute_if_weekend(ts)

def new_years_day_vic(year):
    ts = pd.Timestamp(f"{year}-01-01")
    return substitute_if_weekend(ts)

def australia_day_vic(year):
    ts = pd.Timestamp(f"{year}-01-26")
    return substitute_if_weekend(ts)    

#--- Special Case: ANZAC Day 2011 was substitued to the Tuesday as it fell on Easter Monday ---
def anzac_day_vic(year):
    if year == 2011:
        return pd.Timestamp("2011-04-26")
    return pd.Timestamp(f"{year}-04-25")


#---- Victorian Holiday Dictionary---

HOLIDAYS_VIC = {
    # Holidays with substitute logic
    "New Year's Day": lambda y: new_years_day_vic(y),
    "Australia Day": lambda y: australia_day_vic(y),

    # Victorian-specific holidays
    "Labour Day": lambda y: labour_day_vic(y),
    "King's Birthday": lambda y: kings_birthday_vic(y),
    "Melbourne Cup Day": lambda y: melbourne_cup_vic(y),
    "AFL Grand Final Eve": lambda y: afl_grand_final_eve(y),

    # Easter-related holidays
    "Easter": easter,
    "Good Friday": lambda y: pd.Timestamp(easter(y)) - pd.Timedelta(days=2),
    "Easter Saturday": lambda y: pd.Timestamp(easter(y)) - pd.Timedelta(days=1),
    "Easter Sunday": lambda y: pd.Timestamp(easter(y)),
    "Easter Monday": lambda y: pd.Timestamp(easter(y)) + pd.Timedelta(days=1),

    # ANZAC Day (special case for 2011)
    "ANZAC Day": lambda y: anzac_day_vic(y),

    # Christmas + Boxing Day with substitute logic
    "Christmas Day": lambda y: christmas_day_vic(y),
    "Boxing Day": lambda y: boxing_day_vic(y),
}

#--- Adding Function to generate holiday dates for range of years ---
def build_vic_holiday_dict(start_year, end_year):
    """
    Convert HOLIDAYS_VIC (name → function) into a dictionary:
    year → list of holiday timestamps
    """
    out = {}

    for year in range(start_year, end_year + 1):
        dates = []
        for name, fn in HOLIDAYS_VIC.items():
            ts = fn(year)
            if pd.notna(ts):
                dates.append(ts)
        out[year] = sorted(dates)

    return out

# --- Mapping to group holiday name ---
from VIC_holidays import HOLIDAYS_VIC

def build_holiday_name_map(start_year, end_year):
    """
    Returns: {holiday_name: [Timestamp dates across years]}
    """
    out = {}

    for name, fn in HOLIDAYS_VIC.items():
        dates = []
        for year in range(start_year, end_year + 1):
            ts = fn(year)
            if pd.notna(ts):
                dates.append(ts)
        out[name] = sorted(dates)

    return out

holiday_by_name = build_holiday_name_map(2004, 2018)

