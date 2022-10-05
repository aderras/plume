import single_plume_model as spm
import helpers
from user_inputs import folders

from datetime import datetime
from siphon.simplewebservice.wyoming import WyomingUpperAir
from metpy.units import units
import metpy.calc as mpcalc
import numpy as np
import pandas as pd


def import_ellingson_sounding():

    # Ellingson file details
    ELLINGSON_FILE = folders.DIR_DATA_INPUT+"/tropical_profile_ellingson_250m.txt"
    ELLINGSON_COLUMNS = ["z", "p", "t", "skip1", "skip2", "sh", "rh", "mse",
                        "mse_sat"]
    ELLINGSON_NUM_OF_HEADER_ROWS = 2

    # Reading in the Ellingson sounding
    ellingson_df = pd.read_csv(ELLINGSON_FILE, sep="\s+", header=None,
                                skiprows=ELLINGSON_NUM_OF_HEADER_ROWS)
    ellingson_df.columns = ELLINGSON_COLUMNS
    ellingson_df["z"] = ellingson_df["z"]*1000.0 # Convert to meters

    return ellingson_df

if __name__ == "__main__":

    ellingson_df = import_ellingson_sounding()
    sounding = spm.prep_sounding(ellingson_df.z, ellingson_df.p, ellingson_df.t,
                                ellingson_df.sh)

    sol = spm.run_single_plume(sounding, assume_entr=True)
    spm.save_as_csv(sol)
