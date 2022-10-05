import numpy as np
from user_inputs import constants
import helpers

def compute_buoyancy(t_vc:float, t_va:float, q_w:float) -> float:
    """
                 T_vc - T_va
        B = g * --------------
                  T_va - q_w
    """
    return constants.CONSTANT_G * ((t_vc - t_va)/t_va - q_w)

"""
The following equations are used by the Runge-Kutta solver. The first argument
has to be the independent coordinate and second has to be the dependent one.
"""
def dwcdz(z, w_c, B, ϵ):
    """

        dwc      1
       ------ = ---- * (a_b*B - ϵ*wc^2)
         dz      wc

    """
    return (1.0/w_c)*(constants.a_b*B - ϵ*w_c**2)


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

def sech(x):
    """Numpy does not have a sech function. Defining it here."""
    return 1/np.cosh(x)

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

def dhcdz(z, hc, qi, dqidz, ϵ, ha):
    """

        dhc        dqi
       ----- = Li ----- - ϵ (hc - Li qi - ha)
         dz         dz

    """

    return constants.CONSTANT_LI*dqidz - ϵ*(hc-constants.CONSTANT_LI*qi-ha)

def desatdz(T:float) -> float:
    """
    Derivative of the Claussius Clapyeron equation with respect to T.
    """

    C = T - 273.15 # Convert temperature to celsius

    return 26170.1*np.exp(17.625*C/(C+243.04))/(243.04+C)**2

def dqvcdz(tc, p, dpdz, dtcdz):
    """
    Below, e_c is saturated vapor pressure.

        dq_vc          e_c    dp     1    de_c    dTc
       ------- = -ϵd (------ ---- + --- -------- -----)
          dz           p^2    dz     p    dT_c    dz
    """
    return constants.ϵd*(-compute_esat(tc)*dpdz/p**2 + desatdz(tc)*dtcdz/p)

def compute_qi(tc, qw):
    """
                                  t_c - t_o
        q_i = 0.5*q_w*(1 - tanh(-------------))
                                     d_t
    Temperatures have to be in Kelvin.
    """
    # ang = ((tc - 273.15) - constants.t_o)/constants.d_t
    ang = (tc - constants.t_o)/constants.d_t
    f_i = (1 - np.tanh(ang))/2
    return f_i*qw

def dMdz(wc, dwcdz, ρ, dρdz):
    """

    """
    return ρ*dwcdz + wc*dρdz

def compute_mflux(p, t_vc, wc):
    return compute_density(p, t_vc)*wc

def Mfunc(wc, dwcdz, ρ, dρdz):
    """
        1    d(ρw_c)
      ------ -------
       ρ*w_c   dz
    """
    return 1/wc*dwcdz + 1/ρ*dρdz

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

def compute_Tv(t:float, w:float) -> float:
    """
    Compute virtual temperature given temperature and water vapor mixing ratio.

    in: t = temperature in Kelvin, w = water vapor mixing ratio in kg/kg
    out: tv = virtual temperature in Kelvin
    """
    return t*(1+0.61*w)

def compute_density(p:float, t:float) -> float:
    return p*100/(t*287)

def compute_TC_from_MSE(mse_in, z_in, p_in):
    """
    Compute the saturation (dew point?) temperature from the moist static energy.
    TC = cloud temperature.

    MSE = cp*T + g*z + Lv*q

    in: mse_in = input m
    """

    """
    The following line makes an vector from 100 to 349.99 in steps of 0.01. Why
    are these limits chosen?

    TC will not exceed limits enforced by physics

    JJ is essentially guessing tc values and seeing which one minimizes mse
    """
    tc = np.arange(100,350,0.01) # [K]

    z = np.full(tc.shape,z_in) # [m]
    p = np.full(tc.shape,p_in) # [hPa]
    mse_in = np.full(tc.shape, mse_in)

    mse = compute_mse_sat(tc,z,p)

    diff_mse = np.abs(mse - mse_in)
    ind = np.argmin(diff_mse)

    tc_min = tc[ind]

    return tc_min

def compute_mse(T:float, H:float, P:float, Q:float) -> float:
    """
    Compute the moist static energy from the temperature, height, pressure, and
    specific humidity.

    in: T = temperature in Kelvin, H = height in meters, P = pressure in hPa,
    q = specific density in kg/kg.
    out: mse = moist static energy (joules?)
    """

    Cp = constants.CONSTANT_CP

    mse_internal = Cp*T
    mse_geo = H*constants.CONSTANT_G

    L = constants.CONSTANT_LV

    mse_water = L*Q
    mse = mse_internal + mse_geo  +  mse_water

    return mse

def compute_qsat(t:float, p:float) -> float:
    """
    Compute the saturated mixing ratio using the temperature and pressure.
    Use the function written to compute specific humidity from water vapor
    mixing ratio.

    in: t = temperature in Kelvin, p = pressure in hPa

        q_sat = ϵ * e_sat/p

    (See equation (5.7) of Salby.)

    """
    e_sat = compute_esat(t) # convert e_sat Pascals to hectoPascals
    return 0.622 * e_sat/p

def compute_esat(T:float) -> float:
    """
    Computes the saturation vapor pressure given a tempreature. Clausius-
    Clapyeron Equation.

    in: T = temperature in Kelvin
    out: e_sat = saturation vapor pressure in hPa
    """
    C = T - 273.15 # Convert temperature to celsius
    return 6.1094*np.exp(17.625*C/(C+243.04)) # <-- From wikipedia

def compute_mse_sat(T:float, H:float, P:float) -> float:
    """
    Compute the saturated moist static energy from the temperature, height, and
    pressure.

    in: T = temperature in Kelvin, H = height in meters, P = pressure in hPa,
    out: mse = moist static energy (joules?)
    """
    Cp = constants.CONSTANT_CP

    mse_internal = Cp*T
    mse_geo = H*constants.CONSTANT_G

    # calculate q_sat for given temperature
    qsat = compute_qsat(T,P)
    L = constants.CONSTANT_LV

    mse_water = L*qsat
    mse = mse_internal + mse_geo  +  mse_water

    return mse

def compute_sh_from_q(q):
    return q/(1+q)
