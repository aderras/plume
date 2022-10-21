import single_plume_model as spm
import helpers
from user_inputs import folders
import constants
import weighting_functions as wf

from datetime import datetime
from siphon.simplewebservice.wyoming import WyomingUpperAir
from metpy.units import units
import metpy.calc as mpcalc
import numpy as np
import pandas as pd
import requests
import time

def run_plume(df, fname_suffix, cth=10):
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

    # sol = spm.run_single_plume(sounding, assume_entr=True)
    # helpers.save_dict_elems_as_csv(sol, suffix=fname_suffix)
    #
    # """Get weighted profile"""
    # sol_weighted = wf.get_weighted_profile(sol, sounding, cth=cth)
    # helpers.save_dict_elems_as_csv(sol_weighted,
    #         suffix="_weighted" + fname_suffix)

if __name__ == "__main__":

    # stations_land = ["82532","61052"] # [Manicore, DRRN Niamey-Aero]
    stations_land = ["61052"] # [Manicore, DRRN Niamey-Aero]
    stations_water = []#"47991","94299"] # [RJAM Minamitorishima, Willis Island]

    """Create a list of dates to acquire data"""
    date_start = datetime(2007,7,2,0)
    date_end = datetime(2007,7,7,0)
    dates = pd.date_range(date_start, date_end)
    dates.freq = None

    time_start = datetime.now()

    # df_cth = pd.read_csv(folders.DIR_DATA_STORE + "/cth_61052.csv")
    # df_cth["datetime"] = pd.to_datetime(df_cth["datetime"])

    for station in np.hstack([stations_land, stations_water]):
        for date in dates:

            date_str = date.strftime('%Y%m%d%H')
            fname_suffix = "_" + str(station)  + "_dt" + date_str + \
                           "_dz" + str(constants.Δz)

            df = pd.DataFrame()
            while True:
                """Request data and wait 10 seconds if the server is busy."""
                try:
                    df = WyomingUpperAir.request_data(date, station)
                    break
                except requests.HTTPError as err:
                    print("HTTPError. Waiting ten seconds and trying again.")
                    time.sleep(10)
                except ValueError as err:
                    print(err)
                    print("Skipping ", date, " for station ", station)
                    break
                except IndexError as err:
                    print(err)
                    print("The server returns this when data is unavailable.")
                    print("Skipping ", date, " for station ", station)
                    break

            if len(df)>1:
                print("Starting plume run for date = ", date_str,
                        ", station = ", station)
                # cth_row = df_cth[df_cth["datetime"]==date]
                # cth = cth_row["cth [m]"].values[0]/1e3
                run_plume(df, fname_suffix, cth=10.0)

    time_elapsed = datetime.now() - time_start
    print("Time elapsed (hh:mm:ss.ms) {}".format(time_elapsed))
