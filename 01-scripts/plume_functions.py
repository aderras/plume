import numpy as np
import helpers
from scipy.optimize import fsolve

import numba
from numba import njit
from numba import jit

import constants

@njit()
def compute_z_from_geopotential(Φ, z0=0):
    """
                  1       1
        Φ = G*m*(--- - -------)
                  a      a+z

        Φ = geopotential in m^2/s^2 (given)
        G = 6.73e-11 N m^2/(kg^2) (gravitational constant)
        a = 6.387e6 m (average radius of earth)
        m = 5.975e24 kg (mass of earth)
        z = geometric height in meters

    Solve this equation for z
    """
    return (1/constants.a-Φ/(constants.G*constants.m))**-1 - constants.a

@njit()
def compute_buoyancy(t_vc:float, t_va:float, q_w:float) -> float:
    """
                 T_vc - T_va
        B = g * -------------- - q_w
                    T_va
    """
    return constants.g * ((t_vc - t_va)/t_va - q_w)

@njit()
def compute_mr_from_sh(sh):
    """Compute water vapor mixing ratio from specific humidity """
    return sh/(1. - sh)

"""
The following equations are used by the Runge-Kutta solver. The first argument
has to be the independent coordinate and second has to be the dependent one.
"""
@njit()
def dwcdz(z, w_c, B, ϵ):
    """

        dwc      1
       ------ = ---- * (a_b*B - ϵ*wc^2)
         dz      wc

    """
    return (1.0/w_c)*(constants.a_b*B - ϵ*w_c**2)

@njit()
def dqwdz(z, qw, ϵ, dqvcdz, qvc, qva, wc):
    """

        dqw               dq_vc                      1
       ------ = -ϵ*qw - (------- + ϵ(q_vc-qva) - --------- (qw-qw_crit)*Θ(qw-qw_crit))
         dz                 dz                    τauto*wc
    """
    dqw = -ϵ*qw - (dqvcdz + ϵ*(qvc-qva))
    if qw>constants.q_w_crit:
        return dqw - (1/(constants.τauto*wc))*(qw-constants.q_w_crit)
    return dqw

@njit()
def sech(x):
    """Numpy does not have a sech function. Defining it here."""
    return 1/np.cosh(x)

@njit()
def dqidz(z, qi, qw, dqwdz, Tc, dTcdz):
    """
        dqi           dqw                    wc             dTc
       ------ = 0.5 (------*(1 - tanh(ϕ)) + ---- sech^2(ϕ) -----)
         dz            dz                    dTi             dz

    where
             Tc - Toi
        ϕ = ----------
               dTi
    """
    arg = (Tc - constants.t_o)/constants.d_t
    return 0.5*(dqwdz*(1-np.tanh(arg)) - qw*sech(arg)**2/constants.d_t*dTcdz)

@njit()
def dhcdz(z, hc, qi, dqidz, ϵ, ha):
    """

        dhc        dqi
       ----- = Li ----- - ϵ (hc - Li qi - ha)
         dz         dz

    """
    return constants.Li*dqidz - ϵ*(hc-constants.Li*qi-ha)

@njit()
def desatdT(T:float) -> float:
    """
    Derivative of the Claussius Clapyeron equation with respect to T.
    """
    return constants.Lv*compute_esat(T)/(constants.Rv*T**2)

@njit()
def dqvcdz(tc, p, dpdz, dtcdz):
    """
    Below, e_c is saturated vapor pressure.

        dq_vc          e_c    dp     1    de_c    dTc
       ------- = -ϵd (------ ---- + --- -------- -----)
          dz           p^2    dz     p    dT_c    dz
    """
    return constants.ϵd*(-compute_esat(tc)*dpdz/p**2 + desatdT(tc)*dtcdz/p)

@njit()
def compute_mr_i(tc, qw):
    """
                                  t_c - t_o
        q_i = 0.5*q_w*(1 - tanh(-------------))
                                     d_t
    Temperatures have to be in Kelvin.
    """
    ang = (tc - constants.t_o)/constants.d_t
    f_i = (1 - np.tanh(ang))/2
    return f_i*qw

@njit()
def dMdz(wc, dwcdz, ρ, dρdz):
    """

    """
    return ρ*dwcdz + wc*dρdz

@njit()
def compute_mflux(p, t_vc, wc):
    return compute_density(p, t_vc)*wc

@njit()
def Mfunc(wc, dwcdz, ρ, dρdz):
    """
        1    d(ρw_c)
      ------ -------
       ρ*w_c   dz
    """
    return 1/wc*dwcdz + 1/ρ*dρdz

@njit()
def compute_ϵ_entr(ρ:float, dρdz:float, ϵT:float, wc:float,  B:float) -> float:
    """
    Compute ϵ_dyn by calculating

                1         1   dρ     a_b*B
       ϵ_dyn = --- (ϵT + --- ---- + --------)
                2         ρ   dz      wc^2

    in: ρ = 1D vector of densities, wc = vertical velocity,
    out: a float
    """
    return max(0.0,0.5*(ϵT + (1.0/ρ)*dρdz + constants.a_b*B/wc**2))

@njit()
def compute_Tv(t:float, w:float) -> float:
    """
    Compute virtual temperature given temperature and water vapor mixing ratio.

    in: t = temperature in Kelvin, w = water vapor mixing ratio in kg/kg
    out: tv = virtual temperature in Kelvin
    """
    return t*(1+0.61*w)

@njit()
def compute_density(p:float, t:float) -> float:
    return p*100/(t*287)

@njit()
def compute_TC_from_MSE(mse, z, p):
    """
    Compute the saturation temperature from the moist static energy.
    TC = cloud temperature.

    MSE = cp*T + g*z + Lv*q

    in: mse_in = input m
    """
    x0 = 200
    x1 = 250
    x2 = 300

    max_iter = 50
    tolerance = 1e-5
    steps_taken = 0
    while steps_taken < max_iter and abs(x1-x0) > tolerance:
        fx0 = compute_mse_sat(x0,z,p) - mse
        fx1 = compute_mse_sat(x1,z,p) - mse
        fx2 = compute_mse_sat(x2,z,p) - mse

        L0 = (x0 * fx1 * fx2) / ((fx0 - fx1) * (fx0 - fx2))
        L1 = (x1 * fx0 * fx2) / ((fx1 - fx0) * (fx1 - fx2))
        L2 = (x2 * fx1 * fx0) / ((fx2 - fx0) * (fx2 - fx1))
        new = L0 + L1 + L2
        x0, x1, x2 = new, x0, x1
        steps_taken += 1
    return x0


@njit()
def compute_mse(T:float, H:float, P:float, Q:float) -> float:
    """
    Compute the moist static energy from the temperature, height, pressure, and
    specific humidity.

    in: T = temperature in Kelvin, H = height in meters, P = pressure in hPa,
    q = specific density in kg/kg.
    out: mse = moist static energy (joules?)
    """

    Cp = constants.Cp

    mse_internal = Cp*T
    mse_geo = H*constants.g

    L = constants.Lv

    mse_water = L*Q
    mse = mse_internal + mse_geo  +  mse_water

    return mse

@njit()
def compute_mr_sat(t:float, p:float) -> float:
    """
    Compute the saturated mixing ratio using the temperature and pressure.
    Use the function written to compute specific humidity from water vapor
    mixing ratio.

        q_sat = ϵ * e_sat/(p-e_sat).

    in: t = temperature in Kelvin, p = pressure in hPa
    """
    e_sat = compute_esat(t) # convert e_sat Pascals to hectoPascals
    return 0.622 * e_sat/(p-e_sat)

@njit()
def compute_esat(T:float) -> float:
    """
    Computes the saturation vapor pressure given a tempreature. Clausius-
    Clapyeron Equation.

    in: T = temperature in Kelvin
    out: e_sat = saturation vapor pressure in hPa
    """
    return 2.53e9*np.exp(-5.42e3/T)

@njit()
def compute_mse_sat(T:float, H:float, P:float) -> float:
    """
    Compute the saturated moist static energy from the temperature, height, and
    pressure.

    in: T = temperature in Kelvin, H = height in meters, P = pressure in hPa,
    out: mse = moist static energy (joules?)
    """
    Cp = constants.Cp

    mse_internal = Cp*T
    mse_geo = H*constants.g

    # calculate q_sat for given temperature
    qsat = compute_mr_sat(T,P)
    L = constants.Lv

    mse_water = L*qsat
    mse = mse_internal + mse_geo  +  mse_water

    return mse

@njit()
def compute_sh_from_mr(q):
    return q/(1+q)
