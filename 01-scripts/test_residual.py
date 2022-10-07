import helpers
from user_inputs import constants
from user_inputs import folders
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

import plume_functions

def compute_residual(dydz_func, z:list, y:list, dydz_args:list, frac=False):
    """
    Compute the residual for the numerical result to some differential equation,
        dy
      ------ = f(y,z,...)
        dz
    by computing
        y'(z) - f(y_n, z_n, ...)
    where y'(z) is computed using finite difference.
    """
    dydz_fd = [helpers.ddz(y,i,constants.Δz) for i in range(len(y))]

    f_vec = np.zeros(len(y))
    for i in range(len(y)):
        if y[i]==np.nan:
            f_vec[i] = np.nan
        else:
            dydz_args_i = [x[i] for x in dydz_args]
            f_vec[i] = dydz_func(z[i], y[i], *dydz_args_i)

    if frac:
        return (dydz_fd - f_vec)/dydz_fd
    else:
        return dydz_fd - f_vec

def compute_all_residuals(sol, frac=False):

    all_res = {}
    n_z = sol["w_c"].shape[0]
    n_entr = sol["w_c"].shape[1]

    """Compute the derivatives in z that you will need."""
    dpdz = [helpers.ddz(sol["p"],i,constants.Δz) for i in range(n_z)]

    dtcdz = np.zeros(sol["t_c"].shape)
    dqvcdz_val = np.zeros(sol["t_c"].shape)
    dqidz_val = np.zeros(sol["t_c"].shape)
    dqwdz_val = np.zeros(sol["t_c"].shape)

    for j in range(n_entr):
        for n in range(n_z):
            dtcdz[n,j] = helpers.ddz(sol["t_c"][:,j],n,constants.Δz,"backwards")

            dqvcdz_val[n,j] = plume_functions.dqvcdz(sol["t_c"][n,j],
                                    sol["p"][n], dpdz[n], dtcdz[n,j])
            dqwdz_val[n,j] = plume_functions.dqwdz(sol["z"],
                                    sol["q_w"][n,j], sol["entr"][n,j],
                                    dqvcdz_val[n,j], sol["q_vc"][n,j],
                                    sol["q_va"][n,j], sol["w_c"][n,j])
            dqidz_val[n,j] = plume_functions.dqidz(sol["z"][n],
                                    sol["q_i"][n,j], sol["q_w"][n,j],
                                    dqwdz_val[n,j], sol["t_c"][n,j],
                                    dtcdz[n,j])

    res_dwcdz = np.zeros(sol["w_c"].shape)
    res_dhcdz = np.zeros(sol["w_c"].shape)
    res_dqwdz = np.zeros(sol["w_c"].shape)

    for j in range(n_entr):

        """Computing residual of dw_c/dz equation."""
        res_dwcdz[:,j] = compute_residual(plume_functions.dwcdz, sol["z"],
                            sol["w_c"][:,j], [sol["B"][:,j], sol["entr"][:,j]],
                            frac)


        """Compute the residual of the dh_c/dz equation."""
        res_dhcdz[:,j] = compute_residual(plume_functions.dhcdz, sol["z"],
                            sol["mse_c"][:,j],
                            [sol["q_i"][:,j], dqidz_val[:,j], sol["entr"][:,j],
                            sol["mse_as"]], frac)

        """Compute the residual of the dq_w/dz equation."""
        res_dqwdz[:,j] = compute_residual(plume_functions.dqwdz, sol["z"],
                            sol["q_w"][:,j],
                            [sol["entr"][:,j],dqvcdz_val[:,j],sol["q_vc"][:,j],
                            sol["q_va"][:,j],sol["w_c"][:,j]], frac)

    if frac:
        all_res["res_dwcdz_frac"] = res_dwcdz
        all_res["res_dhcdz_frac"] = res_dhcdz
        all_res["res_dqwdz_frac"] = res_dqwdz
    else:
        all_res["res_dwcdz"] = res_dwcdz
        all_res["res_dhcdz"] = res_dhcdz
        all_res["res_dqwdz"] = res_dqwdz

    return all_res

def save_all_residuals(res, dir=folders.DIR_DATA_OUTPUT):
    helpers.save_dict_elems_as_csv(res, dir)

def import_residuals(dir=folders.DIR_DATA_OUTPUT, frac=False):
    if frac:
        return helpers.import_fnames_as_dict([dir +"/res_dwcdz_frac.csv",
                                              dir +"/res_dhcdz_frac.csv",
                                              dir +"/res_dqwdz_frac.csv"])
    else:
        return helpers.import_fnames_as_dict([dir +"/res_dwcdz.csv",
                                              dir +"/res_dhcdz.csv",
                                              dir +"/res_dqwdz.csv"])

def plot_height_vs_data(data_to_plot, x_label="", save_path=""):

    height = pd.read_csv(folders.DIR_DATA_OUTPUT + "/z.csv")
    entrT = pd.read_csv(folders.DIR_DATA_OUTPUT + "/entrT.csv")

    fig = plt.figure(figsize=(6, 6))

    color = iter(cm.rainbow(np.linspace(0, 1, len(entrT))))
    ax1 = plt.subplot(1,1,1)
    for i, ϵT in enumerate(entrT["0"]):
        c = next(color)
        ax1.plot(data_to_plot[:,i], height, color=c, label=ϵT)
    ax1.set_xscale("log")
    plt.legend(loc=0, title="ϵT")
    plt.xlabel(x_label)
    plt.ylabel('z [m]')

    if save_path!="":
        plt.savefig(save_path)


def plot_residuals(all_res):
    keys = list(all_res.keys())
    plot_height_vs_data(all_res[keys[0]], "Residual of dwc/dz",
                        folders.DIR_FIGS + "/dwcdz_residual.png")
    plot_height_vs_data(all_res[keys[1]], "Residual of dhc/dz",
                        folders.DIR_FIGS + "/dhcdz_residual.png")
    plot_height_vs_data(all_res[keys[2]], "Residual of dqw/dz",
                        folders.DIR_FIGS + "/dqwdz_residual.png")


if __name__ == "__main__":
    ## Begin residual calculations #############################################
    dir = folders.DIR_DATA_OUTPUT

    # fractional = False
    #
    # sol = helpers.import_fnames_as_dict(
    #     [dir + "/z.csv", dir + "/p.csv", dir + "/rho.csv",
    #      dir + "/mse_as.csv", dir + "/w_c.csv", dir + "/t_c.csv",
    #      dir + "/q_w.csv", dir + "/q_vc.csv", dir + "/q_va.csv",
    #      dir + "/w_c.csv", dir + "/q_i.csv", dir + "/mse_c.csv", dir + "/B.csv",
    #      dir + "/entr.csv"])
    # res = compute_all_residuals(sol, frac=fractional)
    # save_all_residuals(res)
    # res_imp = import_residuals(dir, frac=fractional)

    ############################################################################
    # """This chunk of code is used to plot JJ's data"""
    sol0 = helpers.import_fnames_as_dict([dir + "/z.csv",
                                        dir + "/p.csv",
                                        dir + "/mse_as.csv",
                                        dir + "/q_va.csv"
                                        ])

    dir_jj = folders.DIR_DATA_OUTPUT +"_jj"
    sol1 = helpers.import_fnames_as_dict([dir_jj + "/w_c.csv",
                                        dir_jj + "/t_c.csv",
                                        dir_jj + "/q_w.csv",
                                        dir_jj + "/q_vc.csv",
                                        dir_jj + "/q_i.csv",
                                        dir_jj + "/mse_c.csv",
                                        dir_jj + "/B.csv",
                                        dir_jj + "/entr.csv"
                                        ])
    sol = sol0 | sol1
    res = compute_all_residuals(sol)
    save_all_residuals(res, dir_jj)
    res_imp = import_residuals(dir_jj)
    ############################################################################

    plot_residuals(res_imp)
