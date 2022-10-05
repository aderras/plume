import pandas as pd
from user_inputs import *
import numpy as np
import helpers


if __name__ == "__main__":


    height = pd.read_csv(folders.DIR_DATA_OUTPUT + "/height.csv").to_numpy()
    entrT = pd.read_csv(folders.DIR_DATA_OUTPUT + "/entrT.csv").to_numpy()
    height = np.tile(height, (1,len(entrT)))

    dir = folders.DIR_DATA_OUTPUT #+ "_jj"

    sol = helpers.import_fnames_as_dict([dir + "/qi.csv", dir + "/q_w.csv",
                                        dir + "/t_c.csv", dir + "/entr.csv",
                                        dir + "/mse_c.csv", dir + "/B.csv",
                                        dir + "/w_c.csv"])

    qi_div_qw = np.empty(sol["qi"].shape)
    for n in range(sol["qi"].shape[0]):
        for j in range(sol["qi"].shape[1]):
            if sol["qi"][n,j]!=np.nan and sol["q_w"][n,j]!=np.nan and \
                sol["q_w"][n,j]!=0.0:
                qi_div_qw[n,j] = sol["qi"][n,j]/sol["q_w"][n,j]
            else:
                qi_div_qw[n,j] = np.nan

    helpers.plot_y_vs_x(sol["t_c"], qi_div_qw, entrT, "t_c", "qi/q_w", folders.DIR_FIGS+"/qi_div_qw.png")
    helpers.plot_y_vs_x(sol["q_w"], height, entrT, "q_w", "z [m]", folders.DIR_FIGS+"/q_w.png")
    helpers.plot_y_vs_x(sol["qi"], height, entrT, "q_i", "z [m]", folders.DIR_FIGS+"/q_i.png")
    helpers.plot_y_vs_x(sol["entr"], height, entrT, "entr", "z [m]", folders.DIR_FIGS+"/entr.png")
    helpers.plot_y_vs_x(sol["B"], height, entrT, "B", "z [m]", folders.DIR_FIGS+"/B.png")
    helpers.plot_y_vs_x(sol["w_c"], height, entrT, "w_c", "z [m]", folders.DIR_FIGS+"/w_c.png")
    helpers.plot_y_vs_x(sol["t_c"], height, entrT, "t_c", "z [m]", folders.DIR_FIGS+"/t_c.png")
    helpers.plot_y_vs_x(sol["mse_c"], height, entrT, "mse_c", "z [m]", folders.DIR_FIGS+"/mse_c.png", True)
