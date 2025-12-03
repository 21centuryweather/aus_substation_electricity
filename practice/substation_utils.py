# --- Setting up ----

#This gets us into the right directory from home, in order to run the import python script from Mat

#sys = system (module), gives information and control over python interpreter/terminal itself
import sys
sys.path.append("/home/565/pv3484/aus_substation_electricity")

#% is a magic command, special shortcut command that lets you control/interact with notebook environment (ie. gives control of terminal without writing full python code)
%cd aus_substation_electricity_old/

#This section imports the substations that Mat put together

%run /home/565/pv3484/aus_substation_electricity_old/import_substation.py


#    
# ---Creating a function to filter substations by residential fraction range --
import pandas as pd

def filter_residential_fraction(df, lower=0.0, upper=0.2):
    """
    Filter substations by residential fraction range.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with at least 'Name' and 'Residential' columns.
    lower : float, optional
        Lower bound of residential fraction (inclusive).
    upper : float, optional
        Upper bound of residential fraction (inclusive).

    Returns
    -------
    pandas.DataFrame
        DataFrame containing 'Name' and 'Residential' columns
        filtered to the given range.
    """
    mask = (df['Residential'] >= lower) & (df['Residential'] <= upper)
    return df.loc[mask, ['Name', 'Residential']]

