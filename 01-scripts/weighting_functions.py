import numpy as np
import constants
import pickle
from scipy.interpolate import interp1d
from numba import njit

@njit()
def compute_ta(height, temperature, cth):
    """
    Compute the ambient temperature at the cloud top given the cloud top
    height and the sounding height and temperature.

    in: height is a 1D array of floats, temperature is a 1D array of floats,
    cth is a float.

    out: a float representing the ambient temperature at height cth
    """
    z = np.abs(height - cth)
    minval = min(z)
    ind = np.where(z==minval)[0][0]
    ta = temperature[ind]
    return ta

@njit()
def integrate_trapezoidal(f, h, n):
    """
    Compute the discrete integral of a vector f using the trapezoidal rule

    in: f is a 1D array, h is a float representing the horizontal spacing, n is
    an integer representing the length of f.

    out: a float representing the integral of the vector.
    """
    s = 0.0
    s += f[0]/2.0
    for i in range(1, n-1):
        s += f[i]
    s += f[-1]/2.0
    return s * h

@njit()
def compute_weighted_prob(m_ΔT, m_wc, m_z, cth, tc, ta):
    """
    Compute Eq. (21) of ML16.


    """

    Δz = m_z[1]-m_z[0] # Get the horizontal spacing, Δz
    num_z = m_ΔT.shape[0] # Get the number of levels that we consider
    num_entr = m_ΔT.shape[1] # Get the number of levels that we consider

    ############################################################################
    # Begin probability computation (calculating Eq. (21))

    ## Arguments of the exponential function :
    σz = 1.0 # Assume error in z is 1 km.
    σΔT = (1.0/ta) # Assume error in ΔT is 1 Kelvin/temperature at the top
    zt = cth # Cloud top height
    ΔT = (tc-ta)/ta # Normalized pressure difference of at the top of the cloud


    integrand = np.zeros(num_z)
    p_norm = 0.0
    p_arr = np.zeros(num_z)

    # loop through the entrainment rate
    for entr_ind in np.arange(num_entr):

        # Get the vertical velocity and the ΔT for this entrainment rate
        m_wi = m_wc[:, entr_ind]
        m_ΔTi = m_ΔT[:, entr_ind]

        # Determine valid heights as the index where w_c is finite and nonzero
        idx_valid_wc = np.argwhere(~np.isnan(m_wi) & (m_wi != 0.0))
        cb_ind = np.nanmin(idx_valid_wc) # Index of the cloud bottom
        ct_ind = np.nanmax(idx_valid_wc)+1 # Index of the cloud top

        # Compute the integral in Eq. (21) using the trapezoidal rule. First
        # fill the array `integrand` with values of the integrand for all z
        for k in range(num_z): integrand[k]=0.0
        for k in np.arange(cb_ind, ct_ind):
            integrand[k] = np.exp((-(zt - m_z[k])**2)/(2.0*σz**2) - \
                                    (ΔT - m_ΔTi[k])**2/(2.0*σΔT**2))
        intgrl = integrate_trapezoidal(integrand,Δz,num_z)

        # The probability for this entrainment rate is the integral times
        # 1/(length of integration). Store the result in p_arr
        p_arr[entr_ind] = (1.0/(m_z[ct_ind] - m_z[cb_ind]))*intgrl

        # Sum all of the probabilities in order to compute the normalization
        p_norm += p_arr[entr_ind]

    # Normalize all of the probabilites
    return p_arr/p_norm


@njit()
def get_weighted_profile(results, sounding, cth=10.0, ctb=0.019224):

    # Unpack the results
    m_wc, mse_c, mr_w, t_c, B, mflux, m_entrM, m_detrM, m_tva, m_tvc, mr_i, \
            mr_va, mr_vc, m_entr = results

    num_z = m_wc.shape[0] # Get the number of levels that we consider
    num_entr = m_wc.shape[1] # Get the number of levels that we consider

    ############################################################################
    ## Compute the temperature at the top of the cloud

    m_z = sounding[0]/1e3 # Convert sounding height from m to km
    t_sounding = sounding[2]

    # Compute the ambient temperature at the top of the cloud.
    ta = compute_ta(m_z, t_sounding, cth)

    # The cloud top temperature is the ambient temperature plus a funcion of
    # the cloud top buoyancy.
    tc = ta * ctb + ta

    ############################################################################
    ## Compute ΔT = (T_c - T_a)/T_a

    m_ΔT = np.zeros(m_tvc.shape)
    for i in range(num_z):
        for j in range(num_entr):
            if (~np.isnan(m_tvc[i,j])) and (~np.isnan(m_tva[i,j])) and m_tva[i,j]!=0:
                m_ΔT[i, j] = (m_tvc[i, j] - m_tva[i, j])/m_tva[i, j]

    ############################################################################
    ## Compute the weighted probability function for all entrainment rates

    pr = compute_weighted_prob(m_ΔT,m_wc, m_z, cth, tc, ta)

    ############################################################################
    # Compute expected values for the observables of interest
    # to compute another variable:
    #    - create a new array of zeros of length num_z
    #    - add the array to all_obs
    #    - add the observable's corresponding plume model output to all_raw
    mean_w = np.zeros((num_z, ))
    mean_entr = np.zeros((num_z, ))
    mean_detr = np.zeros((num_z, ))
    mean_dt = np.zeros((num_z, ))

    all_obs = [mean_w, mean_entr, mean_detr, mean_dt]
    all_raw = [m_wc, m_entrM, m_detrM, m_ΔT]

    # Sum the probability of
    for i in range(num_entr):

        # finding index of cloud bottom and cloud top, aka have w_c values > 0
        idx_valid_wc = np.argwhere(~np.isnan(m_wc[:, i]) & (m_wc[:, i] != 0.0))
        cb_ind = np.nanmin(idx_valid_wc)
        ct_ind = np.nanmax(idx_valid_wc)+1

        for idobs, obs in enumerate(all_obs):
            obs_raw = all_raw[idobs]
            for idz in range(cb_ind,ct_ind):
                obs[idz] += obs_raw[idz, i]*pr[i]

    # When all of the observables are computed, replace the areas where mean_w
    # is zero or nan with nan.
    for obs in all_obs: obs[(mean_w <=0) ^ (np.isnan(mean_w))] = np.nan

    return (mean_w, mean_dt, mean_entr, mean_detr, pr)
