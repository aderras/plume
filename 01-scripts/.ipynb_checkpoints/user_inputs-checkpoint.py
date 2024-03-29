import os

class constants:

    entrT_list = [0.0, 5e-5, 1e-4, 4e-4, 4e-3]

    CONSTANT_LV = 2.5e6 # [J/kg] Latent heat of vaporization of water
    CONSTANT_LI = 3.34e5 # [J/kg] Latent heat of fusion of water

    CONSTANT_CP = 1005 # [J/(kg.K)] Specific heat a constant pressure of dry air

    CONSTANT_G = 9.81 # [m/s^2] Acceleration of gravity

    Δz = 50. # [m]

    a_b = 1.0/6.0

    τauto = 1.e3 # 10^3s
    q_w_crit = 1.e-3 # 10^-3 kg/kg

    t_o = 253.15 # Kelvin
    d_t = 7.0 # Kelvin

    ϵd = 0.622
    
    fd_scheme = "fe" # OPTIONS: fe=Forward Euler, rk4, rk5

class folders:

    DIR_PAR = os.path.dirname(os.getcwd())
    DIR_DATA = DIR_PAR + "/02-data"
    DIR_DATA_INPUT = DIR_DATA + "/in"
    DIR_DATA_OUTPUT = DIR_DATA + "/out"

    DIR_FIGS = DIR_PAR + "/03-figs"
