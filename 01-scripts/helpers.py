import numpy as np
import pandas as pd
from user_inputs import folders

import constants

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

from numba import njit
from numba import jit

## FINITE DIFFERENCE FUNCTIONS ################################################
@njit()
def fd(dydx, xn:float, yn:float, h:float, dydxArgs:tuple=()):
    if constants.fd_scheme=="fe": return forward_euler(dydx, xn, yn, h, dydxArgs)
    if constants.fd_scheme=="rk4": return rk4(dydx, xn, yn, h, dydxArgs)
    if constants.fd_scheme=="rk5": return rk5(dydx, xn, yn, h, dydxArgs)

@njit()
def forward_euler(dydx, xn:float, yn:float, h:float, dydxArgs:tuple=()):
    k1 = dydx(xn, yn, *dydxArgs)
    return yn + k1*h

"""
Fourth order Runge-Kutta. The following discretizes an equation of the form

        dy
       ---- = f(x, y)
        dx

in: dfdx = function representing the right side of the differential
equation, xn = position at the nth step, yn = value of y at the xn-th step,
dydxArgs (optional) = any other arguments to the function f(x, y).

out: a float representing the y_(n+1) value
"""
@njit()
def rk4(dydx, xn:float, yn:float, h:float, dydxArgs:tuple=()):

    k1 = dydx(xn, yn, *dydxArgs)
    k2 = dydx(xn+0.5*h, yn+0.5*h*k1, *dydxArgs)
    k3 = dydx(xn+0.5*h, yn+0.5*h*k2, *dydxArgs)
    k4 = dydx(xn+h, yn+h*k3, *dydxArgs)

    return yn + (k1+2.0*k2+2.0*k3+k4)*h/6.0

"""Butcher's 5th order RK routine. For reference see pp. 735 of
Numerical Methods for Engineers by Chapra."""
@njit()
def rk5(dydx, xn:float, yn:float, h:float, dydxArgs:tuple=()):

    k1 = dydx(xn, yn, *dydxArgs)
    k2 = dydx(xn + 0.25*h, yn + 0.25*h*k1, *dydxArgs)
    k3 = dydx(xn + 0.25*h, yn + (k1 + k2)*h/8, *dydxArgs)
    k4 = dydx(xn + 0.5*h, yn - h*(0.5*k2 + k3), *dydxArgs)
    k5 = dydx(xn + 0.75*h, yn + h*(3.0*k1+9.0*k4)/16.0, *dydxArgs)
    k6 = dydx(xn + h, yn + h*(-3.0*k1+2.0*k2+12.0*k3-12.0*k4+8.0*k5)/7.0, *dydxArgs)

    return yn + (7.0*k1 + 32.0*k3 + 12.0*k4 + 32.0*k5 + 7.0*k6)*h/90.0


"""
Compute the derivative of a discrete set of points at index n.
"""
# @njit()
def ddz(vec, n, dn, scheme="central"):

    if vec[n]==np.nan: return np.nan

    if n==0 or (n>0 and vec[n-1]==np.nan):
        return (vec[n+1] - vec[n])/dn
    if n==len(vec)-1 or (n<len(vec)-1 and vec[n+1]==np.nan):
        return (vec[n] - vec[n-1])/dn

    if scheme=="forwards":
        return (vec[n+1] - vec[n])/dn
    elif scheme=="backwards":
        return (vec[n] - vec[n-1])/dn
    else:
        return (vec[n+1] - vec[n-1])/(2.0*dn)


## PLOTTING FUNCTIONS #########################################################

def plot_y_vs_x(x_data_to_plot, y_data_to_plot, entrT=[], x_label="",
                y_label="", save_path="", plot_mse_a=[], show_plot=False,
                show_legend=False, xticks_rotation=0, invert_y_axis=False,
                show_grid=False):

    fig = plt.figure(figsize=(6, 6))

    color = iter(cm.rainbow(np.linspace(0, 1, len(entrT))))
    ax1 = plt.subplot(1,1,1)

    if y_data_to_plot.shape != x_data_to_plot.shape:
        y_data_to_plot = np.tile(y_data_to_plot[:,0], (x_data_to_plot.shape[1],1))
        y_data_to_plot = y_data_to_plot.transpose()

    if entrT==[] and x_data_to_plot.shape[1]>1:
        print("No labels given for multi-line plot.")

    if entrT==[]:
        for i in range(x_data_to_plot.shape[1]):
            ax1.plot(x_data_to_plot[:,i], y_data_to_plot[:,i], color="k")
    else:
        for i, ϵT in enumerate(entrT):
            c = next(color)
            ax1.plot(x_data_to_plot[:,i], y_data_to_plot[:,i], color=c, label=ϵT)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(rotation=xticks_rotation)

    if len(plot_mse_a)!=0.0:
        ax1.plot(plot_mse_a[0][:,0], y_data_to_plot[:,0], color="k")
        ax1.plot(plot_mse_a[1][:,0], y_data_to_plot[:,0], color="k", linestyle="--")

    if show_legend: plt.legend(loc=0, title="ϵT")

    if show_grid: plt.grid(linestyle='--', linewidth=0.4)

    if invert_y_axis: ax1.invert_yaxis()

    if save_path!="": plt.savefig(save_path, bbox_inches="tight")

    if show_plot: plt.show()

def plot_row_y_vs_x(x_data_to_plot, y_data_to_plot, entrT=[], x_label="",
                y_label="", save_path="", plot_mse_a=False, same_scale=False,
                show_legend=False, show_grid=False, invert_y_axis=False,
                xticks_rotation=0):

    nplots = len(x_data_to_plot)
    fig = plt.figure(figsize=(5*nplots, 5))

    if y_data_to_plot.shape != x_data_to_plot[0].shape:
        y_data_to_plot = np.tile(y_data_to_plot[:,0], (x_data_to_plot[0].shape[1],1))
        y_data_to_plot = y_data_to_plot.transpose()

    axs = [[] for k in range(nplots)]

    for k in range(nplots):

        color = iter(cm.rainbow(np.linspace(0, 1, len(entrT))))

        axs[k] = plt.subplot(1,nplots,k+1)

        if entrT==[] and x_data_to_plot[k].shape[1]>1:
            print("No labels given for multi-line plot.")

        if entrT==[]:
            for i in range(x_data_to_plot[k].shape[1]):
                axs[k].plot(x_data_to_plot[k][:,i], y_data_to_plot[:,i], color="k")
        else:
            for i, ϵT in enumerate(entrT):
                c = next(color)
                axs[k].plot(x_data_to_plot[k][:,i], y_data_to_plot[:,i], color=c, label=ϵT)

        if show_legend: plt.legend(loc=0, title="ϵT")
        plt.xlabel(x_label[k])
        plt.ylabel(y_label)
        plt.xticks(rotation=xticks_rotation)

        if show_grid: plt.grid(linestyle='--', linewidth=0.4)

        if plot_mse_a:
            mse_a = pd.read_csv(folders.DIR_DATA_OUTPUT + "/mse_a.csv")
            mse_as = pd.read_csv(folders.DIR_DATA_OUTPUT + "/mse_as.csv")
            axs[k].plot(mse_a, y_data_to_plot, color="k")
            axs[k].plot(mse_as, y_data_to_plot, color="k", linestyle="--")

    if same_scale:
        xmin1, xmax1, ymin1, ymax1 = axs[0].axis()
        for k in range(1,nplots):
            xmin2, xmax2, ymin2, ymax2 = axs[k].axis()
            if xmin2<xmin1: xmin1 = xmin2
            if ymin2<ymin1: ymin1 = ymin2
            if xmax2>xmax1: xmax1 = xmax2
            if ymax2>ymax1: ymax1 = ymax2

        for k in range(nplots):
            axs[k].set_xlim(xmin1,xmax1)
            axs[k].set_ylim(ymin1,ymax1)

    for k in range(nplots):
        if invert_y_axis: axs[k].invert_yaxis()

    plt.tight_layout()

    if save_path!="": plt.savefig(save_path, bbox_inches="tight")

def plot_y_vs_x_weighted(x_data_to_plot, y_data_to_plot, y_data_weighted,
        entrT=[], x_label="",
        y_label="",save_path="", plot_mse_a=[], show_plot=False,
        show_legend=False, show_grid=False, invert_y_axis=False,
        xticks_rotation=0):

    fig = plt.figure(figsize=(5, 5))
    ax1 = plt.subplot(1,1,1)


    if y_data_to_plot.shape != x_data_to_plot.shape:
        y_data_to_plot = np.tile(y_data_to_plot[:,0], (x_data_to_plot.shape[1],1))
        y_data_to_plot = y_data_to_plot.transpose()

    if entrT==[]:
        for i in range(x_data_to_plot.shape[1]):
            ax1.plot(x_data_to_plot[:,i], y_data_to_plot[:,i], color="0.8")
    else:
        for i, ϵT in enumerate(entrT):
            ax1.plot(x_data_to_plot[:,i], y_data_to_plot[:,i], color="0.8", label=ϵT)

    ax1.plot(y_data_weighted, y_data_to_plot[:,0], color="k")

    if len(plot_mse_a)!=0.0:
        ax1.plot(plot_mse_a[0], y_data_to_plot, color="y")
        ax1.plot(plot_mse_a[1], y_data_to_plot, color="y", linestyle="--")

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(rotation=xticks_rotation)

    if show_legend: plt.legend(loc=0, title="ϵT")

    if show_grid: plt.grid(linestyle='--', linewidth=0.4)

    if invert_y_axis: ax1.invert_yaxis()

    if save_path!="": plt.savefig(save_path, bbox_inches="tight")

    if show_plot: plt.show()

## IMPORT AND EXPORT FUNCTIONS #################################################

def import_fnames_as_dict(fnames):
    data_dict = {}
    for fn in fnames:
        data_dict[fn.split("/")[-1][:-4]] = pd.read_csv(fn).to_numpy()
    return data_dict

def save_dict_elems_as_csv(dict_of_data, dir=folders.DIR_DATA_OUTPUT, suffix=""):
    for k,v in dict_of_data.items():
        data_df = pd.DataFrame(v)
        data_df.to_csv(dir+"/"+k+suffix+".csv", index=False)

def save_vec_elems_as_csv(vec_of_data, vec_of_names, dir=folders.DIR_DATA_OUTPUT, suffix=""):
    for i,n in enumerate(vec_of_names):
        data_df = pd.DataFrame(vec_of_data[i])
        data_df.to_csv(dir+"/"+n+suffix+".csv", index=False)


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
