# Imports
import numpy as np
from numpy.testing._private.utils import assert_
from scipy.interpolate import interp1d
import metpy
from metpy.units import units

from plume_functions import *
from user_inputs import *
import helpers
# from numba import jit, njit

import pickle

import pandas as pd

"""
CTB = cloud top buoyancy. Is a function of height
"""
# CLIMATOLOGY_CTB = pickle.load(open('./climatology_all.pkl', 'rb'))

def prep_sounding(z:pd.core.series.Series, p:pd.core.series.Series,
                  t:pd.core.series.Series, sh:pd.core.series.Series,
                  mse=None, mse_sat=None):

    """
    Create the sounding array necessary to run the single plume model,
    from individual profiles.

    NOTE: the sounding array is forced to run on a 250m resolution, therefore
    any data provided will be interpolated into a 250m resolution

    in: z = 1D array of heights in km, p = 1D array of pressure in hPa, t =
    1D array of temperature in K, sh = 1D array of specific humidity in kg/kg.
    out:
    """

    # convert all the data into input arrays
    in_z = np.array(z, dtype=float)
    in_z_meters = in_z

    in_p = np.array(p, dtype=float)
    in_sh = np.array(sh, dtype=float)
    in_t = np.array(t, dtype=float)

    # Convert mse and mse_sat to arrays if those are provided
    if (mse is not None): in_mse = np.array(mse, dtype=float)
    if (mse_sat is not None): in_mse_sat = np.array(mse_sat, dtype=float)

    # make sure that all the arrays have the same size
    assert_condition = (in_z.shape == in_p.shape) & \
                        (in_sh.shape == in_t.shape) & \
                        (in_t.shape == in_p.shape)
    assert assert_condition, "All input arrays must have the same dimensions!"

    """Check whether the input z data has desired discretization"""
    if in_z_meters[1]-in_z_meters[0] > constants.Δz:
        print("Input data's resolution of",(in_z[1]-in_z[0]),"does not have"
            " desired resolution. Interpolating to user-specified Δz = ",
            constants.Δz)
        interp_Δz = constants.Δz
    elif in_z_meters[1]-in_z_meters[0] < constants.Δz:
        print("Input data has higher resolution than desired. Skipping"+
        " interpolation.")
        interp_Δz = in_z_meters[1]-in_z_meters[0]



    z_max = 20e3 # Max cloud height is 20 km
    n_z = round(z_max/interp_Δz) # Number of elements in the z direction
    n_obs = 6 # Number of observables: z, p, sh, t, mse_a, mse_as

    # Interpolate the input arrays into the 250 m resolution outputs
    z_meters = np.arange(0, z_max, interp_Δz)

    p_fn_z = interp1d(in_z_meters, in_p, fill_value='extrapolate')
    p = p_fn_z(z_meters)

    t_fn_z = interp1d(in_z_meters, in_t, fill_value='extrapolate')
    t = t_fn_z(z_meters)

    sh_fn_z = interp1d(in_z_meters, in_sh, fill_value='extrapolate')
    sh = sh_fn_z(z_meters)

    if (mse is None):
        mse_a = compute_mse(t, z_meters, p, sh)
    else:
        mse_fn_z = interp1d(in_z_meters, in_mse, fill_value='extrapolate')
        mse_a = mse_fn_z(z_meters)

    if (mse_sat is None):
        mse_as = compute_mse_sat(t, z_meters, p)
    else:
        mse_sat_fn_z = interp1d(in_z_meters, in_mse_sat, fill_value='extrapolate')
        mse_as = mse_sat_fn_z(z)

    """Save sounding data as a dictionary"""
    sounding = {}
    sounding["z"] = z_meters
    sounding["p"] = p
    sounding["t"] = t
    sounding["sh"] = sh
    sounding["mse_a"] = mse_a
    sounding["mse_as"] = mse_as

    return sounding

def find_index_lcl(t, height, p, sh, z_surf):
    # finding the LCL
    i_lcl = -1
    Γd = 9.8/1005.0 # Dry adiabatic laps rate in Kelvin/meter

    s_len = len(height) # Discretization in Z

    # Check that the starting level is above land
    t_start_ind = np.nanmin(np.argwhere(~np.isnan(t) & (height >= z_surf)))

    # Search for the LCL by looping over all of the z values
    for i in np.arange(t_start_ind+1, s_len):
        # air parcel temperature at level if risen by dry adiabatic:
        temp_0 = t[t_start_ind] - Γd*constants.Δz*i

        # saturation mixing ratio for the air parcel, at the given level
        q_sat_0 = compute_qsat(temp_0,p[i])
        if (sh[t_start_ind] > compute_sh_from_q(q_sat_0)):
            # when specific humidity at level 0 exceeds saturation q at
            # level (level 0 is the level at which we have a T value)
            return i
    return -1

def initialize_storage(s_len, e_len):
    # zero'ing out all the variables
    q_w = np.full((s_len,e_len), np.nan) # condensed water
    qi = np.full((s_len,e_len), np.nan) # condensed ice

    B = np.full((s_len,e_len), np.nan) # buoyancy
    w_c = np.full((s_len,e_len), np.nan) # two versions

    entr = np.full((s_len,e_len), np.nan) # entrainment
    detr = np.full((s_len,e_len), np.nan) # detrainment
    mflux = np.full((s_len,e_len), np.nan) # mass flux

    mse_c = np.full((s_len,e_len), np.nan) # _c means inside cloud, _a means ambient

    t_c = np.full((s_len,e_len), np.nan) #

    q_va = np.full((s_len,e_len), np.nan) # vapor phase of q
    q_vc = np.full((s_len,e_len), np.nan) #

    t_va = np.full((s_len,e_len), np.nan) # ambient virtual temperature
    t_vc = np.full((s_len,e_len), np.nan) #

    return [w_c,mse_c,q_w,qi,t_c,B,mflux,entr,detr,q_va,q_vc,t_va,t_vc]

def run_single_plume(sounding, z_surf=0.0, assume_entr=True):

    print("Starting plume run...")

    height = sounding["z"]
    p = sounding["p"]
    t = sounding["t"]
    sh = sounding["sh"]
    mse_a = sounding["mse_a"]
    mse_as = sounding["mse_as"]

    """Calculate density at every height"""
    ρ = [compute_density(p[i],t[i]) for i in range(len(p))]

    entrT_list = constants.entrT_list

    i_lcl = find_index_lcl(t, height, p, sh, z_surf)

    s_len = len(height)
    e_len = len(entrT_list)

    w_c,mse_c,q_w,qi,t_c,B,mflux,entr,detr,\
        q_va,q_vc,t_va,t_vc = initialize_storage(s_len,e_len)

    """Initialize values at the LCL"""
    q_w[i_lcl-1,:] = 0.0
    qi[i_lcl-1,:] = 0.0

    w_c[i_lcl-1,:] = np.sqrt(0.5)
    mse_c[i_lcl-1,:] = mse_as[i_lcl-1]

    for j, entrT in enumerate(entrT_list):
        """
        For all of the entrainment rates, solve the differential
        equations.
        """

        """Start at the LCL and move up"""
        for n in range(i_lcl-1, s_len-1):

            t_c[n,j] = compute_TC_from_MSE(mse_c[n,j], height[n], p[n])
            q_vc[n,j] = compute_qsat(t_c[n,j], p[n])
            q_va[n,j] = compute_qsat(t[n], p[n])

            t_vc[n,j] = compute_Tv(t_c[n,j], q_vc[n,j])
            t_va[n,j] = compute_Tv(t[n], q_va[n,j])

            B[n,j] = compute_buoyancy(t_vc[n,j], t_va[n,j], q_w[n,j])

            mflux[n,j] = compute_mflux(p[n], t_vc[n,j], w_c[n,j])

            qi[n,j] = compute_qi(t_c[n,j], q_w[n,j])

            """
            If this is the LCL, compute entrainment and detrainment
            assuming that dM/dz > 0. Otherwise compute dM/dz to determine
            entrainment and detrainment.
            """
            if n==i_lcl-1 and assume_entr==True:
                dρdz = helpers.ddz(ρ, n, constants.Δz)
                entr[n,j] = compute_ϵ_entr(ρ[n], dρdz, entrT, w_c[n,j], B[n,j])
                detr[n,j] = entrT
            elif n==i_lcl-1 and assume_entr==False:
                entr[n,j] = entrT
            else:
                dmdz = helpers.ddz(mflux[:,j], n, constants.Δz, "backwards")

                if dmdz > 0.0:
                    dρdz = helpers.ddz(ρ, n, constants.Δz)
                    entr[n,j] = compute_ϵ_entr(ρ[n], dρdz, entrT, w_c[n,j], B[n,j])
                    detr[n,j] = entrT
                else:
                    entr[n,j] = entrT
                    detr[n,j] = entrT - dmdz/mflux[n,j]
            ## DEBUGGING ########################################################
            # print("\nStarting loop for level ", n,". Parameters are: ")
            # print("B[n-1,j] = ", B[n-1,j], ",  B[n,j] = ", B[n,j],
            #     "\nmflux[n-1,j] = ", mflux[n-1,j], ",  mflux[n,j] = ", mflux[n,j],
            #     "\nentr[n-1,j] = ", entr[n-1,j], ",  ent[n,j] = ", entr[n,j],
            #     "\nt_va[n-1,j] = ", t_va[n-1,j], ",  t_va[n,j] = ", t_va[n,j],
            #     "\nt_vc[n-1,j] = ", t_vc[n-1,j], ",  t_vc[n,j] = ", t_vc[n,j],
            #     "\ndmdz = ", dmdz
            #     )
            ## END DEBUGGING ####################################################

            """Compute the n+1 element of vertical velocity from the nth one"""
            w_c[n+1,j] = helpers.rk(dwcdz, height[n], w_c[n,j],
                                    constants.Δz, [B[n,j], entr[n,j]])

            if w_c[n+1,j] <= 0.0: break

            dtcdz_param = -0.01 # Arbitrary guess that gets updated
            tol = 1e-6
            err = 10.0
            numloops = 0
            max_loops = 30

            dpdz = helpers.ddz(p,n,constants.Δz)

            ## DEBUGGING #######################################################
            # print("predicted value = ", w_c[n+1,j],
            #         ", dwcdz = ", dwcdz(height[n], w_c[n,j], B[n,j], entr[n,j]),
            #         ", B[n,j] = ", B[n,j],
            #         ", entr = ", entr[n,j],
            #         )
            ## END DEBUGGING ###################################################

            while (np.abs(err) > tol and numloops <= max_loops):

                """Compute dq_vc/dz to be used in dqidz"""
                dqvcdz_val = dqvcdz(t_c[n,j], p[n], dpdz, dtcdz_param)

                dqwdz_val = dqwdz(height[n], q_w[n,j], entr[n,j], dqvcdz_val,
                                q_vc[n,j], q_va[n,j], w_c[n,j])

                dqidz_val = dqidz(height[n], qi[n,j], q_w[n,j], dqwdz_val,
                                t_c[n,j], dtcdz_param)

                """Compute the n+1 element of in-cloud moist static energy"""
                mse_c[n+1,j] = helpers.rk(dhcdz, height[n], mse_c[n,j],
                    constants.Δz, [qi[n,j], dqidz_val, entr[n,j], mse_as[n]])

                """
                Having computed all of the derivatives, we test the dtcdz_param
                is correct by checking that the new temperature from mse_c is
                consistent."""
                Tmse = compute_TC_from_MSE(mse_c[n+1,j], height[n+1], p[n+1])
                dtcdz_temp = (Tmse-t_c[n,j])/constants.Δz
                err = dtcdz_temp - dtcdz_param

                ## DEBUGGING ###################################################
                # if n==140:
                #     print("mse[n+1] = ", mse_c[n+1,j],
                #             ", dqvcdz = ", dqvcdz_val,
                #             ", dqwdz = ", dqwdz_val,
                #             ", dqidz = ", dqidz_val,
                #             ", Tmse = ", Tmse,
                #             ", dtcdz_tmp = ", dtcdz_temp
                #             )
                ## END DEBUGGING ###############################################

                if (np.abs(err) > tol):
                    dtcdz_param = dtcdz_temp
                else:
                    t_c[n+1,j] = Tmse

                numloops += 1
                if numloops == max_loops:
                    print("\nReached maximum loop for entrT = ", entrT,
                            ", level = ", n,
                            ", height = ", n*constants.Δz,
                            ", err = ", err,
                            )

            """Compute the n+1 element of water vapor mixing ratio when the
            while loop converges"""
            q_w[n+1,j] = helpers.rk(dqwdz, height[n], q_w[n,j], constants.Δz,
                            [entr[n,j], dqvcdz_val, q_vc[n,j], q_va[n,j],
                            w_c[n,j]])

            ## DEBUGGING ########################################################
            # print("\nWhile loop ended. Ending parameters for level ", n,"are: ")
            # print("w_c[n] = ",w_c[n,j],", w_c[n+1] = ", w_c[n+1,j],
            #     "\nmse_c[n,j] = ", mse_c[n,j], ",  mse_c[n+1,j] = ", mse_c[n+1,j],
            #     "\nq_w[n,j] = ", q_w[n,j], ",  q_w[n+1,j] = ", q_w[n+1,j],
            #     "\nt_c[n, j] = ", t_c[n,j], ", t_c[n+1,j] = ", t_c[n+1,j],
            #     "\ndtcdz_param = ", dtcdz_param
            #     )
            ## END DEBUGGING ####################################################

        if n<s_len: w_c[n+1,j]=np.nan

    return {"w_c":w_c, "mse_c":mse_c, "q_w":q_w, "t_c":t_c, "B":B,
            "mflux": mflux, "entr":entr, "detr":detr, "t_va":t_va,
            "t_vc":t_vc, "qi":qi, "q_va":q_va, "q_vc":q_vc, "height":height,
            "entrT":entrT_list, "rho":ρ}

def save_as_csv(dict_of_data):

    for k,v in dict_of_data.items():
        data_df = pd.DataFrame(v)
        data_df.to_csv(folders.DIR_DATA_OUTPUT+"/"+k+".csv", index=False)
