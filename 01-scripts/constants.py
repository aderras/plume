import numpy as np

entrT_list = np.append(np.arange(0, 0.1, 0.01), [0.1, 0.12, 0.15, 0.2, 0.3, 0.4])/1000
# entrT_list = np.array([0.001])

Lv = 2.5e6 # [J/kg] Latent heat of vaporization of water
Li = 3.34e5 # [J/kg] Latent heat of fusion of water

Rv = 461.5 # [J/(kg*K)] specific gas constant of water vapor

Cp = 1005 # [J/(kg.K)] Specific heat a constant pressure of dry air

g = 9.81 # [m/s^2] Acceleration of gravity

Δz = 50. # [m]

a_b = 1.0/6.0

τauto = 1.e3 # 10^3s
q_w_crit = 1.e-3 # 10^-3 kg/kg

t_o = 253.15 # Kelvin
d_t = 7.0 # Kelvin

ϵd = 0.622

fd_scheme = "fe" # OPTIONS: fe=Forward Euler, rk4, rk5

w_c_init = 1.5

dtcdz_tol = 6.5e-5

G = 6.73e-11 # [N m^2/(kg^2)] (gravitational constant)
a = 6.387e6 # [m] (average radius of earth)
m = 5.975e24 # [kg] (mass of earth)
