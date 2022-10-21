import pandas as pd
from user_inputs import *
import numpy as np
import helpers
from datetime import datetime


if __name__ == "__main__":

    date_choice = 0

    ## The following has to match what is in run_plume_wyoming.py ##############
    stations_land = ["82532"]#,"61052"] # [Manicore, DRRN Niamey-Aero]
    stations_water = []#"47991","94299"] # [RJAM Minamitorishima, Willis Island]

    station = stations_land[0]

    """Create a list of dates to acquire data"""
    date_start = datetime(2022,7,1,12)
    date_end = datetime(2022,7,30,12)
    dates = pd.date_range(date_start, date_end)
    ############################################################################

    """Check that file exists before continuing."""

    dir = folders.DIR_DATA_OUTPUT
    date_str = dates[date_choice].strftime('%Y%m%d%H')
    fname_suffix = "_" + str(station)  + "_dt" + date_str + \
                   "_dz" + str(constants.Δz)


    obs_list = ["entr", "w_c", "q_w", "mse_c", "B", "z", "w_c_weighted",
                "entr_weighted", "entrT",  "mse_a",  "mse_as", "p", "t", "sh"]
    fnames_list = [dir + "/" + x + fname_suffix + ".csv" for x in obs_list]

    sol = helpers.import_fnames_as_dict(fnames_list)

    entrT_rounded = np.round(sol["entrT"+fname_suffix],decimals=7)

    """Plot sounding"""
    helpers.plot_y_vs_x(sol["p"+fname_suffix], sol["z"+fname_suffix],
        show_plot=True,
        x_label="p (hPa)", y_label="z (m)",
        # save_path=folders.DIR_FIGS + "/p"+fname_suffix+".png"
        )
    helpers.plot_y_vs_x(sol["t"+fname_suffix], sol["z"+fname_suffix],
        show_plot=True,
        x_label="T (K)", y_label="z (m)",
        # save_path=folders.DIR_FIGS + "/t"+fname_suffix+".png"
        )
    helpers.plot_y_vs_x(sol["sh"+fname_suffix], sol["z"+fname_suffix],
        show_plot=True,
        x_label="sh", y_label="z (m)",
        # save_path=folders.DIR_FIGS + "/sh"+fname_suffix+".png"
        )

    """Plot results"""
    helpers.plot_y_vs_x(sol["q_w"+fname_suffix], sol["z"+fname_suffix],
        show_plot=True,
        entrT=entrT_rounded, x_label="q_w", y_label="z (m)", show_legend=True,
        # save_path=folders.DIR_FIGS + "/q_w"+fname_suffix+".png"
        )
    helpers.plot_y_vs_x(sol["w_c"+fname_suffix], sol["z"+fname_suffix],
        show_plot=True,
        show_legend=True,
        entrT=entrT_rounded, x_label="w_c (m/s)", y_label="z (m)",
        # save_path=folders.DIR_FIGS + "/w_c"+fname_suffix+".png"
        )
    helpers.plot_y_vs_x(sol["entr"+fname_suffix], sol["z"+fname_suffix],
        show_plot=True,
        show_legend=True,
        entrT=entrT_rounded, x_label="entr (1/m)", y_label="z (m)",
        # save_path=folders.DIR_FIGS + "/entr"+fname_suffix+".png"
        )
    helpers.plot_y_vs_x(sol["B"+fname_suffix], sol["z"+fname_suffix],
        show_plot=True,
        show_legend=True,
        entrT=entrT_rounded, x_label="B (N/kg)", y_label="z (m)",
        # save_path=folders.DIR_FIGS + "/B"+fname_suffix+".png"
        )
    helpers.plot_y_vs_x(sol["mse_c"+fname_suffix], sol["z"+fname_suffix],
        plot_mse_a = [sol["mse_a"+fname_suffix],sol["mse_as"+fname_suffix]],
        show_legend=True,
        show_plot=True,
        entrT=entrT_rounded, x_label="mse", y_label="height",
        # save_path=folders.DIR_FIGS + "/mse_c"+fname_suffix+".png"
        )

    """Plot weighted outputs"""
    helpers.plot_y_vs_x_weighted(sol["w_c"+fname_suffix],
        sol["z"+fname_suffix], sol["w_c_weighted"+fname_suffix],
        x_label="w_c_weighted", y_label="z (m)", invert_y_axis=False,
        show_plot=True,
        # save_path=folders.DIR_FIGS+"/w_c_weighted"+fname_suffix+".png"
        )
    helpers.plot_y_vs_x_weighted(sol["entr"+fname_suffix],
        sol["z"+fname_suffix], sol["entr_weighted"+fname_suffix],
        x_label="entr_weighted", y_label="z (m)",
        xticks_rotation = 45,
        show_plot=True,
        # save_path=folders.DIR_FIGS+"/entr_weighted"+fname_suffix+".png"
        )
