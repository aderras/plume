import pandas as pd
from user_inputs import *
import numpy as np
import helpers

if __name__ == "__main__":


    dir = folders.DIR_DATA_OUTPUT

    sol = helpers.import_fnames_as_dict([dir + "/q_i.csv", dir + "/q_w.csv",
                                        dir + "/t_c.csv", dir + "/entr.csv",
                                        dir + "/mse_c.csv", dir + "/B.csv",
                                        dir + "/w_c.csv", dir + "/entrT.csv",
                                        dir + "/w_c_weighted.csv",
                                        dir + "/z.csv", dir + "/p.csv",
                                        dir + "/entr_weighted.csv"])

    # qi_div_qw = np.empty(sol["q_i"].shape)
    # for n in range(sol["q_i"].shape[0]):
    #     for j in range(sol["q_i"].shape[1]):
    #         if sol["q_i"][n,j]!=np.nan and sol["q_w"][n,j]!=np.nan and \
    #             sol["q_w"][n,j]!=0.0:
    #             qi_div_qw[n,j] = sol["q_i"][n,j]/sol["q_w"][n,j]
    #         else:
    #             qi_div_qw[n,j] = np.nan

    # helpers.plot_y_vs_x(sol["t_c"], qi_div_qw, entrT, "t_c", "qi/q_w", folders.DIR_FIGS+"/qi_div_qw.png")
    # helpers.plot_y_vs_x(sol["q_w"], height, entrT, "q_w", "z [m]", folders.DIR_FIGS+"/q_w.png")
    # helpers.plot_y_vs_x(sol["q_i"], height, entrT, "q_i", "z [m]", folders.DIR_FIGS+"/q_i.png")
    # helpers.plot_y_vs_x(sol["entr"], height, entrT, "entr", "z [m]", folders.DIR_FIGS+"/entr.png")
    # helpers.plot_y_vs_x(sol["B"], height, entrT, "B", "z [m]", folders.DIR_FIGS+"/B.png")
    # helpers.plot_y_vs_x(sol["w_c"], height, entrT, "w_c", "z [m]", folders.DIR_FIGS+"/w_c.png")
    # helpers.plot_y_vs_x(sol["t_c"], height, entrT, "t_c", "z [m]", folders.DIR_FIGS+"/t_c.png")

    # helpers.plot_y_vs_x(sol["mse_c"], height, entrT, "mse_c", "z [m]",
    #     plot_mse_a=[sounding["mse_a"],sounding["mse_as"]] folders.DIR_FIGS+"/mse_c.png")

    helpers.plot_y_vs_x_weighted(sol["w_c"], sol["p"], sol["w_c_weighted"],
        x_label="w_c_weighted", y_label="p (hPa)", invert_y_axis=True,
        show_plot=True, save_path=folders.DIR_FIGS+"/w_c_weighted.png")

    helpers.plot_y_vs_x_weighted(sol["entr"], sol["p"], sol["entr_weighted"],
        x_label="entr_weighted", y_label="p (hPa)", invert_y_axis=True,
        xticks_rotation = 45, show_plot=True,
        save_path=folders.DIR_FIGS+"/entr_weighted.png")
