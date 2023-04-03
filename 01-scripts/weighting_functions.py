import numpy as np
import constants
import pickle
from scipy.interpolate import interp1d
from numba import njit
import helpers


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
def compute_weighted_prob(m_ΔT, m_wc, m_z, cth, tc, ta):
    """
    This function evaluates Eq. (21) of ML16.

    p_likelihood = p(z_T, ΔT_T | ϵ_{tur,j})
    p_priori = p(ϵ_{tur,j})

    """

    Δz = m_z[1]-m_z[0] # Get the horizontal spacing, Δz
    num_z = m_ΔT.shape[0] # Get the number of levels that we consider
    num_entr = m_ΔT.shape[1] # Get the number of entrainments that we consider

    ############################################################################
    # Begin probability computation (calculating Eq. (21))

    ## Arguments of the exponential function:
    σz = 1.0 # Assume error in z is 1 km.
    σΔT = (1.0/ta) # Assume error in ΔT is 1/temperature at the top
    zt = cth # Cloud top height
    ΔT = (tc-ta)/ta # Normalized pressure difference of at the top of the cloud

    p_norm = 0.0 # Normalization constant for the probability

    # Initialize temporary storage
    integrand = np.zeros(num_z) # The integrand
    m_dΔTidz = np.zeros(num_z)
    m_wi = np.zeros(num_z)
    m_ΔTi = np.zeros(num_z)

    p_likelihood = np.zeros(num_entr)
    p_posteriori = np.zeros(num_entr)
    p_priori = np.zeros(num_entr)

    integral = 0.0

    # loop through the entrainment rate
    for entr_ind in np.arange(num_entr):

        for k in range(num_z):
            # Get the vertical velocity and the ΔT for this entrainment rate
            m_wi[k] = m_wc[k, entr_ind]
            m_ΔTi[k] = m_ΔT[k, entr_ind]

            # Reset storage to zero
            integrand[k] = 0.0
            m_dΔTidz[k] = 0.0

        # Determine valid heights as the index where w_c is finite and nonzero
        idx_valid_wc = np.argwhere(~np.isnan(m_wi) & (m_wi != 0.0))
        cb_ind = np.nanmin(idx_valid_wc) # Index of the cloud bottom
        ct_ind = np.nanmax(idx_valid_wc)+1 # Index of the cloud top

        # Compute the derivative of ΔT w/r/t z to use in the line integral. The
        # line we integrate over is the ΔT vs. z trajectory.
        for k in range(num_z): m_dΔTidz[k] = helpers.ddz(m_ΔTi,k,Δz)

        # Compute the integral in Eq. (21) using the trapezoidal rule. First
        # fill the array `integrand` with values of the integrand for all z.
        # Then integrate using the trapezoidal rule.
        for k in np.arange(cb_ind, ct_ind):
            integrand[k] = np.exp(-0.5*((zt - m_z[k])/σz)**2 - \
                                   0.5*((ΔT - m_ΔTi[k])/σΔT)**2) * \
                                   np.sqrt(m_dΔTidz[k]**2 + 1)
        integral = helpers.integrate_trapezoidal(integrand,Δz,num_z)

        # The probability for this entrainment rate is the integral times
        # 1/(length of integration). Store the result in p_likelihood
        p_likelihood[entr_ind] = (1.0/(m_ΔTi[ct_ind] - m_ΔTi[cb_ind]))*integral

        # Sum all of the probabilities in order to compute the normalization
        p_norm += p_likelihood[entr_ind]

        #########################################################################
        ##  DEBUGGING
        # print("entr_ind = ", entr_ind, ", ϵ = ", constants.entrT_list[entr_ind],
        #     "\nfirst exp arg = ", (zt - m_z)/σz,
        #     "\nsecond exp arg = ", ((ΔT - m_ΔTi)/σΔT), # "\ndΔTdz = ", m_dΔTidz,
        #     "\nIntegral = ", integral, ", p_cond = ", p_likelihood[entr_ind])
        #########################################################################

    # Have to make sure probability is normalized
    p_likelihood = p_likelihood/p_norm

    ############################################################################
    # Compute the posteriori probability. Assuming that the priori probability
    # for ϵ is uniform. Normalize the result
    p_priori = np.ones(num_entr)/num_entr

    ## Test whether setting the priori to a gaussian centered at the median
    ## entrainment rate changes the results.
    # xvals = np.arange(0,1,1/num_entr)
    # for k in range(num_entr): p_priori[k] = helpers.gaussian(xvals[k], 0.5, 1)

    # The posterior probability is the product of the prior and the likelihood
    p_posteriori = p_priori*p_likelihood
    p_posteriori = p_posteriori/(np.nansum(p_posteriori))

    ############################################################################
    ## DEGUGGING
    # print("p(ϵ) = ",p_priori)
    # print("p(z_T, ΔT_t | ϵ_{tur,j}) = ",p_likelihood)
    # return p_likelihood
    ############################################################################

    return p_posteriori

@njit()
def get_weighted_profile(results, sounding, cth=10.0, ctb=0.019224):

    # Unpack the results
    m_wc, mse_c, mr_w, t_c, B, mflux, m_entrM, m_detrM, m_tva, m_tvc, mr_i, \
            mr_va, mr_vc, m_qcond, m_qauto, m_entr = results

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
    # Compute the weighted probability function for all entrainment rates
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
    mean_qcond = np.zeros((num_z, ))
    mean_qauto = np.zeros((num_z, ))


    all_obs = [mean_w, mean_entr, mean_detr, mean_dt, mean_qcond, mean_qauto]
    all_raw = [m_wc, m_entrM, m_detrM, m_ΔT, m_qcond, m_qauto]

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

    return (mean_w, mean_dt, mean_entr, mean_detr, pr, mean_qcond, mean_qauto)
