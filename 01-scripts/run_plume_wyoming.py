import single_plume_model as spm
import helpers
from user_inputs import folders
from user_inputs import constants
import weighting_functions as wf

from datetime import datetime
from siphon.simplewebservice.wyoming import WyomingUpperAir
from metpy.units import units
import metpy.calc as mpcalc
import numpy as np
import time
import pandas as pd
import requests

def run_plume(df, fname_suffix):
    P  = df['pressure'].values * units(df.units['pressure'])
    T  = df['temperature'].values * units(df.units['temperature'])
    Z  = df['height'].values * units(df.units['height'])
    Td = df['dewpoint'].values * units(df.units['dewpoint'])
    SH = mpcalc.specific_humidity_from_dewpoint(P,Td)

    P2 = np.array(P)
    T2 = np.array(T) + 273.15 # Convert to Kelvin
    Z2 = np.array(Z)
    SH2 = SH

    sounding = spm.prep_sounding(Z2, P2, T2, SH2)
    helpers.save_dict_elems_as_csv(sounding, suffix=fname_suffix)

    sol = spm.run_single_plume(sounding, assume_entr=True)
    helpers.save_dict_elems_as_csv(sol, suffix=fname_suffix)

    """Get weighted profile"""
    sol_weighted = wf.get_weighted_profile(sol, sounding, cth=10)
    helpers.save_dict_elems_as_csv(sol_weighted,
            suffix="_weighted" + fname_suffix)

if __name__ == "__main__":

    stations_land = ["82532","61052"] # [Manicore, DRRN Niamey-Aero]
    stations_water = ["47991","94299"] # [RJAM Minamitorishima, Willis Island]

    station = stations_land[1]
    # station = stations_water[0]

    """Create a list of dates to acquire data"""
    date_start = datetime(2022,9,1,0)
    date_end = datetime(2022,9,30,0)
    dates = pd.date_range(date_start, date_end)
    # dates = pd.date_range(date_start, date_end, freq="W")

    for date in dates:

        date_str = date.strftime('%Y%m%d%H')
        fname_suffix = "_" + str(station)  + "_dt" + date_str + \
                       "_dz" + str(constants.Δz)

        df = pd.DataFrame()
        while True:
            """Request data and wait 10 seconds if the server is busy."""
            try:
                df = WyomingUpperAir.request_data(date,station)
                break
            except requests.HTTPError as err:
                print("HTTPError. Waiting ten seconds and trying again.")
                time.sleep(10)
            except ValueError as err:
                print(err)
                print("Skipping this.")
                break

        if len(df)!=0:
            print("Starting plume run for date = ", date_str,
                    ", station = ", station)
            run_plume(df, fname_suffix)
