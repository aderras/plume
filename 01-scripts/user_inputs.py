import os
import numpy as np

class era5_options:

    # The following constants are used for the ERA5 data:
    time = np.arange(0.0, 2400.0, 100.0)
    level = np.flip(np.array([1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150,\
                      175, 200, 225, 250, 300, 350, 400, 450, 500, 550,\
                      600, 650, 700, 750, 775, 800, 825, 850, 875, 900,\
                      925, 950, 975, 1000]))
    lat = np.arange(0,721)
    lon = np.arange(0,1440)

class folders:

    DIR_PAR = os.path.dirname(os.getcwd())
    DIR_DATA = DIR_PAR + "/02-data"
    DIR_DATA_INPUT = DIR_DATA + "/in"
    DIR_DATA_OUTPUT = DIR_DATA + "/out"
    DIR_DATA_POSTPROC = DIR_DATA + "/post-proc"
    DIR_DATA_STORE = DIR_DATA + "/store"

    DIR_FIGS = DIR_PAR + "/03-figs"
