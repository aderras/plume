import os

# set number of CPUs to run on
ncore = "6"

os.environ["OMP_NUM_THREADS"] = ncore
os.environ["OPENBLAS_NUM_THREADS"] = ncore
os.environ["MKL_NUM_THREADS"] = ncore
os.environ["VECLIB_MAXIMUM_THREADS"] = ncore
os.environ["NUMEXPR_NUM_THREADS"] = ncore

import single_plume_model as spm
import helpers
import plume_functions
from user_inputs import folders
import constants
from user_inputs import era5_options
import weighting_functions as wf

from datetime import datetime
import pygrib
import numpy as np
import pandas as pd
import multiprocessing
import itertools

def get_single_col_data(indx_era5, ti=0, lat_index=0, lon_index=0):

    n_levels = len(era5_options.level)

    gp_col = np.zeros(n_levels)
    t_col = np.zeros(n_levels)
    sh_col = np.zeros(n_levels)

    for i, lev in enumerate(era5_options.level):
        msg = indx_era5.select(level=float(lev), typeOfLevel="isobaricInhPa",
                          parameterName="Temperature",
                          time=float(era5_options.time[ti]))
        t_col[i] = msg[0].values[lat_index, lon_index]

        msg = indx_era5.select(level=float(lev), typeOfLevel="isobaricInhPa",
                          parameterName="Geopotential",
                          time=float(era5_options.time[ti]))
        gp_col[i] = msg[0].values[lat_index, lon_index]

        msg = indx_era5.select(level=float(lev), typeOfLevel="isobaricInhPa",
                          parameterName="Specific humidity",
                          time=float(era5_options.time[ti]))
        sh_col[i] = msg[0].values[lat_index, lon_index]


    return {"t":t_col, "sh":sh_col, "gp":gp_col}

def compute_wc_max(args):

    time_choice, lat_choice, lon_choice = args

    fname_era5 = folders.DIR_DATA_INPUT + "/era5.grib"

    indx_era5 = pygrib.index(fname_era5, "typeOfLevel", "level",
                          "parameterName", "time")
    print("Latitude = ", lat_choice, ", longitude = ", lon_choice)

    inp = get_single_col_data(indx_era5, time_choice, lat_choice,
                              lon_choice)

    z = [plume_functions.compute_z_from_geopotential(x) \
         for x in inp["gp"]]
    inp["z"] = z
    inp["p"] = era5_options.level # Pressure in hPa

    sounding = spm.prep_sounding(inp["z"], inp["p"], inp["t"], inp["sh"])
    sol = spm.run_single_plume(sounding, assume_entr=True)
    # Skip saving because it takes too much space
    # helpers.save_dict_elems_as_csv(sol)
    # helpers.save_dict_elems_as_csv(sounding)

    sol_weighted = wf.get_weighted_profile(sol, sounding, cth=10)
    # helpers.save_dict_elems_as_csv(sol_weighted, suffix="_weighted")

    """Determine w_c_max from the weighted solution"""
    return np.nanmax(sol_weighted["w_c"])

if __name__ == "__main__":

    time_start = datetime.now()

    # fname_era5 = folders.DIR_DATA_INPUT + "/era5.grib"

    ############################################################################
    ## Use the following code snippet to print information about the grib file
    ## that you can use to create an index object. This object is used to
    ## quickly access data in a large file.
    # data_era5 = pygrib.open(fname_era5)
    # all_levels = []
    # for i, g in enumerate(data_era5):
    #     print(g.shortName, g.typeOfLevel, g.level, g.time)
    #     if g.level not in all_levels: all_levels.append(g.level)
    #     if i==0: print(g)
    # data_era5.close()
    # print(all_levels)
    ############################################################################

    # indx_era5 = pygrib.index(fname_era5, "typeOfLevel", "level",
    #                         "parameterName", "time")
    # lat_choice = 300 # OPTIONS: Integer in range(0,721)
    # lon_choice = 700 # OPTIONS: Integer in range(0,1440)
    time_choice = 0 # OPTIONS: Integer in range(0,24)

    """Initialize storage for max(w_c)"""
    w_c_max = np.full((len(era5_options.lat),len(era5_options.lon)), np.nan)

    for lat_choice in era5_options.lat[::10]:
        time_elapsed = datetime.now() - time_start
        print("\nMoving to next latitude. Time elapsed (hh:mm:ss.ms) {}".format(time_elapsed))
        with multiprocessing.Pool(6) as p:
            wcmax = p.map(compute_wc_max, zip(itertools.repeat(time_choice),
                    itertools.repeat(lat_choice),era5_options.lon[::10]))
            w_c_max[lat_choice,::10] = wcmax
            pd.DataFrame(w_c_max).to_csv(folders.DIR_DATA_POSTPROC+"/w_c_max.csv",
                                         index=False)

    time_elapsed = datetime.now() - time_start
    print("Time elapsed (hh:mm:ss.ms) {}".format(time_elapsed))
