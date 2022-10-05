import numpy as np
import pandas as pd
from user_inputs import folders

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm


"""
Select Runge-Kutta scheme by uncommenting one of the lines in this function.
"""
def rk(dydx, xn:float, yn:float, h:float, dydxArgs:list=[]):
    # return rk4(dydx, xn, yn, h, dydxArgs)
    return rk5(dydx, xn, yn, h, dydxArgs)
    # return forward_euler(dydx, xn, yn, h, dydxArgs)

def forward_euler(dydx, xn:float, yn:float, h:float, dydxArgs:list=[]):
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
def rk4(dydx, xn:float, yn:float, h:float, dydxArgs:list=[]):

    k1 = dydx(xn, yn, *dydxArgs)
    k2 = dydx(xn+0.5*h, yn+0.5*h*k1, *dydxArgs)
    k3 = dydx(xn+0.5*h, yn+0.5*h*k2, *dydxArgs)
    k4 = dydx(xn+h, yn+h*k3, *dydxArgs)

    return yn + (k1+2.0*k2+2.0*k3+k4)*h/6.0

"""Butcher's 5th order RK routine. For reference see pp. 735 of
Numerical Methods for Engineers by Chapra."""
def rk5(dydx, xn:float, yn:float, h:float, dydxArgs:list=[]):

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
def ddz(vec, n, dn, scheme="central"):

    if vec[n]==np.nan: return np.nan

    if n==0 or (n>0 and vec[n-1]==np.nan): scheme="forwards"
    if n==len(vec)-1 or (n<len(vec)-1 and vec[n+1]==np.nan): scheme="backwards"

    if scheme=="forwards":
        return (vec[n+1] - vec[n])/dn
    elif scheme=="backwards":
        if n>1 and vec[n-2]==np.nan:
            return (3.0*vec[n]-4.0*vec[n-1]+vec[n-2])/(2.0*dn)
        else:
            return (vec[n] - vec[n-1])/dn
    else:
        return (vec[n+1] - vec[n-1])/(2.0*dn)

def import_fnames_as_dict(fnames):
    data_dict = {}
    for fn in fnames:
        data_dict[fn.split("/")[-1][:-4]] = pd.read_csv(fn).to_numpy()
    return data_dict

def save_dict_elems_as_csv(dict_of_data, dir=folders.DIR_DATA_OUTPUT):
    for k,v in dict_of_data.items():
        data_df = pd.DataFrame(v)
        data_df.to_csv(dir+"/"+k+".csv", index=False)


def plot_y_vs_x(x_data_to_plot, y_data_to_plot, entrT, x_label="", y_label="",
                save_path="", plot_mse_a=False):

    fig = plt.figure(figsize=(6, 6))

    color = iter(cm.rainbow(np.linspace(0, 1, len(entrT))))
    ax1 = plt.subplot(1,1,1)
    for i, ϵT in enumerate(entrT):
        c = next(color)
        ax1.plot(x_data_to_plot[:,i], y_data_to_plot[:,i], color=c, label=ϵT)
    plt.legend(loc=0, title="ϵT")
    # ax1.set_yscale("log")
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if plot_mse_a:
        mse_a = pd.read_csv(folders.DIR_DATA_OUTPUT + "/mse_a.csv")
        mse_as = pd.read_csv(folders.DIR_DATA_OUTPUT + "/mse_as.csv")
        ax1.plot(mse_a, y_data_to_plot, color="k")
        ax1.plot(mse_as, y_data_to_plot, color="k", linestyle="--")


    if save_path!="":
        plt.savefig(save_path)

    # plt.show()
