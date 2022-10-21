import pandas as pd
from user_inputs import *
import numpy as np
import helpers

if __name__ == "__main__":


    dir = folders.DIR_DATA_OUTPUT

    fname_suffix=""
    sol = helpers.import_fnames_as_dict([dir + "/q_i.csv", dir + "/q_w.csv",
                                        dir + "/t_c.csv", dir + "/entr.csv",
                                        dir + "/mse_c.csv", dir + "/B.csv",
                                        dir + "/w_c.csv", dir + "/entrT.csv",
                                        dir + "/w_c_weighted.csv",
                                        dir + "/z.csv", dir + "/p.csv",
                                        dir + "/entr_weighted.csv",
                                        dir + "/t.csv", dir + "/sh.csv",
                                        dir + "/mse_a.csv", dir + "/mse_as.csv"])

    entrT_rounded = np.round(sol["entrT"],decimals=7)

    # Plot sounding data
    helpers.plot_y_vs_x(sol["z"], sol["p"], x_label="z (m)", y_label="p (hPa)",
            show_plot=True,
            invert_y_axis=True,
            # save_path=folders.DIR_FIGS+"/z_vs_p.png"
            )
    helpers.plot_y_vs_x(sol["t"], sol["p"], x_label="T (K)", y_label="p (hPa)",
            show_plot=True,
            invert_y_axis=True,
            # save_path=folders.DIR_FIGS+"/t_vs_p.png"
            )
    helpers.plot_y_vs_x(sol["sh"], sol["p"], x_label="sh", y_label="p (hPa)",
            show_plot=True,
            invert_y_axis=True,
            # save_path=folders.DIR_FIGS+"/sh_vs_p.png"
            )


    qi_div_qw = np.empty(sol["q_i"].shape)
    for n in range(sol["q_i"].shape[0]):
        for j in range(sol["q_i"].shape[1]):
            if sol["q_i"][n,j]!=np.nan and sol["q_w"][n,j]!=np.nan and \
                sol["q_w"][n,j]!=0.0:
                qi_div_qw[n,j] = sol["q_i"][n,j]/sol["q_w"][n,j]
            else:
                qi_div_qw[n,j] = np.nan

    helpers.plot_y_vs_x(sol["t_c"], qi_div_qw, entrT=entrT_rounded,
            x_label="t_c", y_label="qi/q_w",
            show_plot=True,
            # save_path=folders.DIR_FIGS+"/qi_div_qw.png"
            )
    helpers.plot_y_vs_x(sol["entr"], sol["p"], entrT=entrT_rounded,
            x_label="entrT", y_label="p (hPa)",
            invert_y_axis=True,
            xticks_rotation=45,
            show_plot=True,
            # save_path=folders.DIR_FIGS+"/entr_vs_p.png"
            )
    helpers.plot_y_vs_x(sol["q_w"], sol["p"], entrT=entrT_rounded,
            x_label="q_w", y_label="p (hPa)",
            invert_y_axis=True,
            show_plot=True,
            # save_path=folders.DIR_FIGS+"/qw_vs_p.png"
            )
    helpers.plot_y_vs_x(sol["B"], sol["p"], entrT=entrT_rounded,
            x_label="B [N/kg]", y_label="p (hPa)",
            invert_y_axis=True,
            show_plot=True,
            # save_path=folders.DIR_FIGS+"/B_vs_p.png"
            )
    helpers.plot_y_vs_x(sol["w_c"], sol["p"], entrT=entrT_rounded,
            x_label="w_c", y_label="p (hPa)",
            invert_y_axis=True,
            show_plot=True,
            # save_path=folders.DIR_FIGS+"/wc_vs_p.png"
            )

    helpers.plot_y_vs_x(sol["mse_c"], sol["p"], entrT=entrT_rounded,
            x_label="mse_c [J/kg]", y_label="p (hPa)",
            plot_mse_a=[sol["mse_a"],sol["mse_as"]],
            xticks_rotation=45,
            show_plot=True,
            invert_y_axis=True,
            # save_path=folders.DIR_FIGS+"/mse_c.png"
            )
    """Plot weighted outputs"""
    helpers.plot_y_vs_x_weighted(sol["w_c"+fname_suffix],
        sol["z"+fname_suffix], sol["w_c_weighted"+fname_suffix],
        x_label="w_c_weighted", y_label="z (m)", invert_y_axis=False,
        show_plot=True,
        # save_path=folders.DIR_FIGS+"/w_c_weighted"+fname_suffix+".png"
        )
    helpers.plot_y_vs_x_weighted(sol["entr"+fname_suffix],
        sol["p"+fname_suffix], sol["entr_weighted"+fname_suffix],
        x_label="entr_weighted", y_label="z (m)",
        xticks_rotation = 45,
        show_plot=True,
        invert_y_axis=True,
        # save_path=folders.DIR_FIGS+"/entr_weighted"+fname_suffix+".png"
        )
