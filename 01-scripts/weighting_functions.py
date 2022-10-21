import numpy as np
import constants
import pickle
from scipy.interpolate import interp1d


def import_climatology_ctb(in_cth=10.0):
    """Getting a climatological CTB given a CTH"""
    CLIMATOLOGY_CTB = pickle.load(open('/home/amel/Documents/plume-model/02-data/in/climatology_all.pkl','rb'))

    cth = CLIMATOLOGY_CTB['cth']
    ctb = CLIMATOLOGY_CTB['ctb']

    f = interp1d(cth, ctb)

    try:
        out_ctb = f(in_cth)
    except ValueError:
        return np.nan

    return out_ctb

def compute_ta(height, temperature, cth):
    z = np.abs(height - cth)
    ind = np.nanargmin(z)
    ta = temperature[ind]
    return ta

def get_weighted_profile(in_dict, sounding, cth=10.0):
    ''' Wrapper code to run the weighting of SPM_outputs given the spm_output_dictionary
    input:
        single plume model output from cm.run_single_plume(), Should be a dictionary
        cth -- height in km (cth)
        tc -- cloud top temperature in K
        ta -- ambient temperature in K
    '''

    ctb = import_climatology_ctb(cth)
    ta = compute_ta(sounding["z"], sounding["t"], cth)
    tc = ta * ctb + ta

    zt = cth
    dtt = (tc-ta)/ta

    m_w = in_dict["w_c"]
    m_tvc = in_dict["t_vc"]
    m_tva = in_dict["t_va"]
    m_entrM = in_dict["entr"]
    m_detrM = in_dict["detr"]
    m_entr = in_dict["entrT"]

    m_z = sounding["z"]/1e3 # Have to convert to km

    # makding sure the division is only done for values that are valid
    valid_ind = (~np.isnan(m_tvc)) & (~np.isnan(m_tva)) & ~(m_tva == 0.)
    m_dt = np.zeros(m_tvc.shape)

    for i in range(m_dt.shape[0]):
        for j in range(m_dt.shape[1]):
            if (valid_ind[i, j]):
                m_dt[i, j] = (m_tvc[i, j] - m_tva[i, j])/m_tva[i, j]
            if (np.isnan(m_entrM[i, j])):
                m_entrM[i, j] = 0.
            if (np.isnan(m_detrM[i, j])):
                m_detrM[i, j] = 0.

    # computing sigma T -- measurement errors
    sigma_dt = 2. * (1./ta)**2
    sigma_z = 2. * (1.)**2

    # setting zeros for final weighted profile
    mean_w = np.zeros(m_w.shape[0])
    mean_dt = np.zeros(m_w.shape[0])
    mean_entr = np.zeros(m_w.shape[0])
    mean_detr = np.zeros(m_w.shape[0])

    # probabilty initialization to zeros
    p_norm = 0
    p_arr = np.zeros(m_entr.shape[0])

    h_cond_grid = np.zeros((m_z.shape[0], m_entr.shape[0]))

    # loop through the entrainment rate
    for entr_ind in np.arange(0, m_entr.shape[0]):

        # getting the model profiles for a given entr rate
        # entr_val = m_entr[entr_ind]
        m_wi = m_w[:, entr_ind]
        m_dti = m_dt[:, entr_ind]

        # finding out where w_c goes to zero or vanishes
        ind = np.argwhere(~np.isnan(m_wi) & (m_wi != 0.0))

        # finding index of cloud bottom and cloud top, aka have w_c values > 0
        cb_ind = np.nanmin(ind)
        ct_ind = np.nanmax(ind)+1
        # print("cb_ind = ",cb_ind, ", ct_ind = ", ct_ind)

        # looping through the cloud top and bottom
        temp_sum = 0
        h_cond_grid[cb_ind:ct_ind, entr_ind] = 1
        for k in np.arange(cb_ind, ct_ind):

            # computing the integration from cloud bottom to cloud top, eq 21
            e_z = np.exp((-1.*(zt - m_z[k])**2)/(sigma_z))
            e_dt = np.exp((-1.*(dtt - m_dti[k])**2)/(sigma_dt))
            temp_sum += e_z * e_dt

        # final prob of obs for a given entr rate
        p_obs_entr = (1./(ct_ind - cb_ind)) * (temp_sum)

        p_arr[entr_ind] = p_obs_entr

        # sum up the prob of all obs given entr
        p_norm += p_obs_entr

    cnt = 0
    pr = np.ones(m_entr.shape[0])/m_entr.shape[0]
    p_arr = p_arr/p_norm

    while (cnt < 100):
        pr = pr * p_arr
        pr = pr / np.nansum(pr)
        cnt = cnt + 1

    mean_w = np.zeros((m_z.shape[0], ))
    mean_entr = np.zeros((m_z.shape[0], ))
    mean_detr = np.zeros((m_z.shape[0], ))
    mean_dt = np.zeros((m_z.shape[0], ))

    for i in range(m_w.shape[1]):

        # finding out where w_c goes to zero or vanishes
        ind = np.argwhere(~np.isnan(m_w[:, i]) & (m_w[:, i] != 0.0))

        # finding index of cloud bottom and cloud top, aka have w_c values > 0
        cb_ind = np.nanmin(ind)
        ct_ind = np.nanmax(ind)+1

        for n in range(cb_ind, ct_ind):
            mean_w[n] += m_w[n, i] * pr[i] * h_cond_grid[n, i]
            mean_entr[n] += m_entrM[n, i] * pr[i] * h_cond_grid[n, i]
            mean_detr[n] += m_detrM[n, i] * pr[i] * h_cond_grid[n, i]
            mean_dt[n] += m_dt[n, i] * pr[i] * h_cond_grid[n, i]

    mean_entr[mean_w <= 0.0] = np.nan
    mean_entr[mean_w == np.nan] = np.nan
    mean_detr[mean_w <= 0.0] = np.nan
    mean_detr[mean_w == np.nan] = np.nan
    mean_dt[mean_w <= 0.0] = np.nan
    mean_dt[mean_w == np.nan] = np.nan
    mean_w[mean_w <= 0.0] = np.nan

    return {'w_c': mean_w, 'dt': mean_dt, 'entr': mean_entr, 'detr': mean_detr,
            'pr': pr, 'pr_orig': p_arr, 'h_cond': h_cond_grid}
