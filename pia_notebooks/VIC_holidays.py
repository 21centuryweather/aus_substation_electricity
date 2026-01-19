__title__: "Holiday Dictionary for Victoria"
__purpose__: "Create a dictionary of all National and Victorian public holidays so I can call upon it in notebooks rather than copying and pasting the library each time"

__Victorian_hoiday_list:
# Labour Day is observed on the second Monday in March
# Melbourne Cup is observed on the First Tuesday in November
# AFL Grand Final (25/09/15, 30/09/16, 29/09/17)
# King's Birthday is observed on the second Monday in June

#--- Helper Function to calculate Labour Day, Melb Cup and King's Birthday ---

import pandas as pd
from datetime import date, timedelta

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

def afl_grand_final_eve(year):
    """
    AFL Grand Final Eve public holiday:
    - The Grand Final is held on the last Saturday of September.
    - The public holiday is the Friday immediately before.
    """
    # Find last Saturday of September
    last_day = date(year, 9, 30)
    offset = (last_day.weekday() - 5) % 7  # Saturday = 5
    grand_final_day = last_day - timedelta(days=offset)
    grand_final_eve = grand_final_day - timedelta(days=1)
    return pd.Timestamp(grand_final_eve)

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

#--- Christmas and Boxing Day substitute logic ---
def christmas_day_vic(year):
    ts = pd.Timestamp(f"{year}-12-25")
    return substitute_if_weekend(ts)

def boxing_day_vic(year):
    ts = pd.Timestamp(f"{year}-12-26")
    return substitute_if_weekend(ts)

#--- Special Case: ANZAC Day 2011 was substitued to the Tuesday as it fell on Easter Monday ---
def anzac_day_vic(year):
    if year == 2011:
        return pd.Timestamp("2011-04-26")
    return pd.Timestamp(f"{year}-04-25")


#---- Victorian Holiday Dictionary---

HOLIDAYS_VIC = {
    "New Year's Day": lambda y: pd.Timestamp(f"{y}-01-01"),
    "Australia Day": lambda y: pd.Timestamp(f"{y}-01-26"),

    # Victorian-specific holidays
    "Labour Day": lambda y: labour_day_vic(y),
    "King's Birthday": lambda y: kings_birthday_vic(y),
    "Melbourne Cup Day": lambda y: melbourne_cup_vic(y),
    "AFL Grand Final Eve": lambda y: afl_grand_final_eve(y),

    # Easter-related holidays
    "Good Friday": lambda y: pd.Timestamp(easter(y)) - pd.Timedelta(days=2),
    "Easter Saturday": lambda y: pd.Timestamp(easter(y)) - pd.Timedelta(days=1),
    "Easter Sunday": lambda y: pd.Timestamp(easter(y)),
    "Easter Monday": lambda y: pd.Timestamp(easter(y)) + pd.Timedelta(days=1),

    # Holidays with special substitute logic
    "ANZAC Day": lambda y: anzac_day_vic(y),
    "Christmas Day": lambda y: christmas_day_vic(y),
    "Boxing Day": lambda y: boxing_day_vic(y),
}



