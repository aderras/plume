import pandas as pd
import matplotlib.pyplot as plt
from user_inputs import *
from matplotlib.pyplot import cm
import numpy as np
import helpers
import plume_functions


"""
This module contains functions that test the consistency of equations used in
the plume program. In order to run this, you must run `run_plume.py` so that
data is output to the 02-data directory.
"""

def compute_dwcdz_and_plot():
    """
    Compute dw_c/dz by
        1. Calculating the finite difference derivative of the solution output
        2. Using the `dwcdz` function obtained by rewriting Eq. (11) of ML16

    This function plots the results and outputs figures.
    """
    dir = folders.DIR_DATA_OUTPUT

    height = pd.read_csv(dir + "/height.csv").to_numpy()
    entrT = pd.read_csv(dir + "/entrT.csv").to_numpy()
    height = np.tile(height, (1,len(entrT)))

    sol = helpers.import_fnames_as_dict([dir + "/entr.csv", dir + "/B.csv",
                                        dir + "/w_c.csv"])

    dwcdz_fd = np.full(sol["w_c"].shape, np.nan)
    dwcdz_fun = np.full(sol["w_c"].shape, np.nan)

    n_z = sol["w_c"].shape[0]
    n_entr = sol["w_c"].shape[1]


    for j in range(n_entr):
        for n in range(n_z-200):

            if sol["w_c"][n,j]!=np.nan and sol["w_c"][n,j]!=0.0:
                """
                Use backwards difference for the following because in the plume
                program that's what you would have to use."""
                dwcdz_fd[n,j] = helpers.ddz(sol["w_c"][:,j],n,constants.Δz, "backwards")
                dwcdz_fun[n,j] = plume_functions.dwcdz(0.0, sol["w_c"][n,j], sol["B"][n,j], sol["entr"][n,j])


    helpers.plot_y_vs_x(dwcdz_fd, height, entrT, "dwc/dz Finite Difference", "z [m]", folders.DIR_FIGS+"/test_dwcdz_fd.png")
    helpers.plot_y_vs_x(dwcdz_fun, height, entrT, "dwc/dz Function", "z [m]", folders.DIR_FIGS+"/test_dwcdz_fun.png")


def compute_entrs_and_plot():
    """
    Compute entraintment rate when dM/dz > 0 using
        1. The analytical expression for ϵ derived from ML16
        2. An alternative analytical expression, Mfunc, which uses the
           finite difference derivative of w_c.
        3. The finite difference derivative of the mass flux.

    This function plots the results and outputs figures.
    """
    height = pd.read_csv(folders.DIR_DATA_OUTPUT + "/height.csv").to_numpy()
    entrT = pd.read_csv(folders.DIR_DATA_OUTPUT + "/entrT.csv").to_numpy()
    height = np.tile(height, (1,len(entrT)))

    dir = folders.DIR_DATA_OUTPUT

    sol = helpers.import_fnames_as_dict([dir + "/rho.csv", dir + "/B.csv",
                                        dir + "/w_c.csv", dir + "/p.csv",
                                        dir + "/t_vc.csv", dir + "/mflux.csv"])

    sol["rho"] = [x[0] for x in sol["rho"]]

    dρdz = np.full(sol["w_c"].shape, np.nan)
    dwcdz = np.full(sol["w_c"].shape, np.nan)
    dMdz = np.full(sol["w_c"].shape, np.nan)

    n_z = sol["w_c"].shape[0]
    n_entr = sol["w_c"].shape[1]

    i_lcl = 3

    for j in range(n_entr):
        for n in range(i_lcl-1,n_z):
            dρdz[n,j] = helpers.ddz(sol["rho"],n,constants.Δz)

            if sol["mflux"][n,j]!=np.nan and sol["mflux"][n,j] !=0.0:
                """
                Use backwards difference for the following because in the plume
                program that's what you would have to use."""
                dwcdz[n,j] = helpers.ddz(sol["w_c"][:,j],n,constants.Δz, "backwards")
                dMdz[n,j] = helpers.ddz(sol["mflux"][:,j],n,constants.Δz,"backwards")

    entr0 = np.full(sol["w_c"].shape, np.nan)
    entr1 = np.full(sol["w_c"].shape, np.nan)
    entr2 = np.full(sol["w_c"].shape, np.nan)

    for j in range(n_entr):
        for n in range(i_lcl-1,n_z):

            if dMdz[n,j] > 0.0 and sol["w_c"][n,j]!=np.nan:

                """Compute entrainment rate using the equation derived from dwc/dz"""
                entr0[n,j] = plume_functions.compute_ϵ_entr(sol["rho"][n],
                            dρdz[n,j], entrT[j], sol["w_c"][n,j], sol["B"][n,j])

                if sol["mflux"][n,j]!=0.0 and sol["mflux"][n,j]!=np.nan:
                    """Compute entrainment rate using Mfunc"""
                    entr1[n,j] = entrT[j] + plume_functions.Mfunc(sol["w_c"][n,j], dwcdz[n,j], sol["rho"][n], dρdz[n,j])/sol["mflux"][n,j]

                    """Compute entrainment using the FD derivative of mflux"""
                    entr2[n,j] = entrT[j] + dMdz[n,j]/sol["mflux"][n,j]

            elif dMdz[n,j] < 0.0 and sol["w_c"][n,j]!=np.nan:
                entr0[n,j] = entrT[j]
                entr1[n,j] = entrT[j]
                entr2[n,j] = entrT[j]

    helpers.plot_y_vs_x(entr0, height, entrT, "entr", "z [m]", folders.DIR_FIGS+"/test_entr0.png")
    helpers.plot_y_vs_x(entr1, height, entrT, "entr", "z [m]", folders.DIR_FIGS+"/test_entr1.png")
    helpers.plot_y_vs_x(entr2, height, entrT, "entr", "z [m]", folders.DIR_FIGS+"/test_entr2.png")


if __name__ == "__main__":
    compute_entrs_and_plot()
    compute_dwcdz_and_plot()
