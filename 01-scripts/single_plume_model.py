# Imports
import numpy as np
from numpy.testing._private.utils import assert_
from scipy.interpolate import interp1d
import pandas as pd

from plume_functions import *
from user_inputs import *
import helpers

from datetime import datetime


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

    interp_Δz = constants.Δz

    z_max = 20e3 # Max cloud height is 20 km
    n_z = round(z_max/interp_Δz) # Number of elements in the z direction

    # Interpolate the input arrays into the 250 m resolution outputs
    z_meters = np.arange(0, z_max, interp_Δz)

    p_fn_z = interp1d(in_z_meters, in_p, fill_value='extrapolate')
    p = p_fn_z(z_meters)

    t_fn_z = interp1d(in_z_meters, in_t, fill_value='extrapolate')
    t = t_fn_z(z_meters)

    sh_fn_z = interp1d(in_z_meters, in_sh, fill_value='extrapolate')
    sh = sh_fn_z(z_meters)

    if (mse is not None):
        mse_fn_z = interp1d(in_z_meters, in_mse, fill_value='extrapolate')
        mse_a = mse_fn_z(z_meters)

    if (mse_sat is not None):
        mse_sat_fn_z = interp1d(in_z_meters, in_mse_sat, fill_value='extrapolate')
        mse_as = mse_sat_fn_z(z)

    if (mse is None):
        mse_a = compute_mse(t, z_meters, p, sh)

    if (mse_sat is None):
        mse_as = compute_mse_sat(t, z_meters, p)

    ρ = np.zeros(n_z)
    dρdz = np.zeros(n_z)
    dpdz = np.zeros(n_z)

    """Calculate density at every height"""
    for i in range(len(ρ)): ρ[i] = compute_density(p[i],t[i])
    for i in range(len(dρdz)): dρdz[i] = helpers.ddz(ρ,i,interp_Δz)
    for i in range(len(dpdz)): dpdz[i] = helpers.ddz(p,i,interp_Δz)

    return (z_meters, p, t, sh, mse_a, mse_as, ρ, dρdz, dpdz)

@njit()
def find_index_lcl(t, height, p, sh, z_surf):
    # finding the LCL
    i_lcl = -1
    Γd = 9.8/1005.0 # Dry adiabatic laps rate in Kelvin/meter
    Δz = height[1]-height[0]

    s_len = len(height) # Discretization in Z

    # Check that the starting level is above land
    t_start_ind = np.nanmin(np.argwhere(~np.isnan(t) & (height >= z_surf)))

    # Search for the LCL by looping over all of the z values
    for i in np.arange(t_start_ind+1, s_len):
        # air parcel temperature at level if risen by dry adiabatic:
        temp_0 = t[t_start_ind] - Γd*Δz*i

        # saturation mixing ratio for the air parcel, at the given level
        mr_sat_0 = compute_mr_sat(temp_0,p[i])
        if (sh[t_start_ind] > compute_sh_from_mr(mr_sat_0)):
            # when specific humidity at level 0 exceeds saturation sh at
            # level
            return i
    return -1

@njit()
def initialize_storage(s_len, e_len):
    # zero'ing out all the variables
    mr_w = np.full((s_len,e_len), np.nan) # condensed water
    mr_i = np.full((s_len,e_len), np.nan) # condensed ice

    B = np.full((s_len,e_len), np.nan) # buoyancy
    w_c = np.full((s_len,e_len), np.nan) # two versions

    entr = np.full((s_len,e_len), np.nan) # entrainment
    detr = np.full((s_len,e_len), np.nan) # detrainment
    mflux = np.full((s_len,e_len), np.nan) # mass flux

    mse_c = np.full((s_len,e_len), np.nan) # _c means inside cloud, _a means ambient

    t_c = np.full((s_len,e_len), np.nan) #

    mr_va = np.full((s_len,e_len), np.nan) # vapor phase of q
    mr_vc = np.full((s_len,e_len), np.nan) #

    t_va = np.full((s_len,e_len), np.nan) # ambient virtual temperature
    t_vc = np.full((s_len,e_len), np.nan) #

    mr_cond = np.full((s_len,e_len), np.nan) # ambient virtual temperature
    mr_auto = np.full((s_len,e_len), np.nan) #

    return (w_c, mse_c, mr_w, mr_i, t_c, B, mflux, entr, detr, mr_va, mr_vc,
            t_va, t_vc, mr_cond, mr_auto)

@njit()
def run_single_plume(storage, sounding, z_surf=0.0, assume_entr=True):

    # print("Starting plume run...")

    w_c, mse_c, mr_w, mr_i, t_c, B, mflux, entr, detr, mr_va, mr_vc, t_va, \
            t_vc, mr_cond, mr_auto = storage
    height, p, t, sh, mse_a, mse_as, ρ, dρdz, dpdz = sounding
    Δz = height[1]-height[0]
    """
    p = pressure [hPa]
    t = temperature [K]
    sh = specific humidity [kg/kg]
    ρ = density [kg/m^3]
    height = height [m]

    w_c = in-cloud vertical velocity [m/s]

    mse_a = ambient moist static energy [J/kg]
    mse_c = in-cloud moist static enregy [J/kg]

    mr_w = water vapor mixing ratio
    mr_i = ice mixing ratio
    mr_va = ambient water vapor mixing ratio
    mr_vc = in-cloud saturated water vapor mixing ratio

    t_c = in-cloud temperature [K]
    t_vc = in-cloud virtual temperature [K]
    t_va = atmosperic virtual temperature [K]

    B = buoyancy [N/kg] = [m/s^2]

    """
    entrT_list = constants.entrT_list

    i_lcl = find_index_lcl(t, height, p, sh, z_surf)
    assert i_lcl != -1
    # print("Found i_lcl = ", i_lcl, ", height of ", height[i_lcl])

    s_len = len(height)
    e_len = len(entrT_list)

    """Initialize values at the LCL"""
    mr_w[i_lcl-1,:] = 0.0
    mr_i[i_lcl-1,:] = 0.0
    w_c[i_lcl-1,:] = constants.w_c_init

    # Initial moist-static-energy is saturated atmospheric moist-static-energy
    mse_c[i_lcl-1,:] = mse_as[i_lcl-1]

    for j, entrT in enumerate(entrT_list):
        """
        For all of the entrainment rates, solve the differential
        equations.
        """

        """Start at the LCL and move up"""
        for n in range(i_lcl-1, s_len-1):

            t_c[n,j] = compute_TC_from_MSE(mse_c[n,j], height[n], p[n])
            mr_vc[n,j] = compute_mr_sat(t_c[n,j], p[n])
            mr_va[n,j] = compute_mr_from_sh(sh[n])
            t_vc[n,j] = compute_Tv(t_c[n,j], mr_vc[n,j])
            t_va[n,j] = compute_Tv(t[n], mr_va[n,j])
            B[n,j] = compute_buoyancy(t_vc[n,j], t_va[n,j], mr_w[n,j])

            mflux[n,j] = compute_mflux(p[n], t_vc[n,j], w_c[n,j])

            mr_i[n,j] = compute_mr_i(t_c[n,j], mr_w[n,j])

            """
            If this is right below the LCL, compute entrainment and detrainment
            assuming that dM/dz > 0. Otherwise compute dM/dz to determine
            entrainment and detrainment.
            """
            if n==i_lcl-1 and assume_entr==True:
                entr[n,j] = compute_ϵ_entr(ρ[n], dρdz[n], entrT, w_c[n,j], B[n,j])
                detr[n,j] = entrT
            elif n==i_lcl-1 and assume_entr==False:
                entr[n,j] = entrT
                detr[n,j] = entrT
            else:
                dmdz = helpers.ddz(mflux[:,j], n, Δz, "backwards")

                if dmdz > 0.0:
                    entr[n,j] = compute_ϵ_entr(ρ[n], dρdz[n], entrT, w_c[n,j], B[n,j])
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
            w_c[n+1,j] = helpers.fd(dwcdz, height[n], w_c[n,j],
                                    Δz, (B[n,j], entr[n,j]))

            if (w_c[n+1,j] <= 0.0 or np.isnan(w_c[n+1,j])): break

            dtcdz_param = -0.01 # Arbitrary guess that gets updated
            tol = constants.dtcdz_tol
            err = 10.0
            numloops = 0
            max_loops = 30

            ## DEBUGGING #######################################################
            # print("predicted value = ", w_c[n+1,j],
            #         ", dwcdz = ", dwcdz(height[n], w_c[n,j], B[n,j], entr[n,j]),
            #         ", B[n,j] = ", B[n,j],
            #         ", entr = ", entr[n,j],
            #         )
            ## END DEBUGGING ###################################################


            dqvcdz_val = dqwdz_val = dqidz_val = dqcdt_val = dqadt_val = 0

            while 1:
                # time_start = datetime.now()

                """Compute dq_vc/dz to be used in dqidz"""
                dqvcdz_val = dqvcdz(t_c[n,j], p[n], dpdz[n], dtcdz_param)

                dqcdt_val = dqcdt(w_c[n,j], dqvcdz_val, entr[n,j], mr_vc[n,j],
                                mr_va[n,j])

                dqadt_val = dqadt(mr_w[n,j])

                # dqwdz_val = dqwdz(height[n], mr_w[n,j], entr[n,j], dqvcdz_val,
                #                 mr_vc[n,j], mr_va[n,j], w_c[n,j])

                dqwdz_val = -entr[n,j]*mr_w[n,j] + (1/w_c[n,j])*(dqcdt_val-dqadt_val)

                dqidz_val = dqidz(height[n], mr_i[n,j], mr_w[n,j], dqwdz_val,
                                t_c[n,j], dtcdz_param)

                """Compute the n+1 element of in-cloud moist static energy"""
                mse_c[n+1,j] = helpers.fd(dhcdz, height[n], mse_c[n,j],
                    Δz, (mr_i[n,j], dqidz_val, entr[n,j], mse_a[n]))

                """
                Having computed all of the derivatives, we test the dtcdz_param
                is correct by checking that the new temperature from mse_c is
                consistent."""

                Tmse = compute_TC_from_MSE(mse_c[n+1,j], height[n+1], p[n+1])
                dtcdz_temp = (Tmse-t_c[n,j])/Δz
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

                # print("t_while-loop = ", (datetime.now() - time_start).total_seconds())

                if (np.abs(err) > tol):
                    dtcdz_param = dtcdz_temp
                else:
                    t_c[n+1,j] = Tmse
                    break

                numloops += 1
                if numloops == max_loops:
                    print("\nReached maximum loop for entrT = ", entrT,
                            ", level = ", n,
                            ", height = ", n*Δz,
                            ", err = ", err,
                            )
                    # Replace all the wc values with np.nan when the solution
                    # fails.
                    for ii in np.arange(len(w_c)): w_c[ii] = np.nan

                    break

            """Compute the n+1 element of water vapor mixing ratio when the
            while loop converges"""
            mr_w[n+1,j] = helpers.fd(dqwdz, height[n], mr_w[n,j], Δz,
                            (entr[n,j], dqvcdz_val, mr_vc[n,j], mr_va[n,j],
                            w_c[n,j]))


            """Store the condensation rate and the autoconversion rate"""
            mr_cond[n+1,j] = dqcdt_val
            mr_auto[n+1,j] = dqadt_val

            ## DEBUGGING ########################################################
            # print("\nWhile loop ended. Ending parameters for level ", n,"are: ")
            # print("w_c[n] = ",w_c[n,j],", w_c[n+1] = ", w_c[n+1,j],
            #     "\nmse_c[n,j] = ", mse_c[n,j], ",  mse_c[n+1,j] = ", mse_c[n+1,j],
            #     "\nmr_w[n,j] = ", mr_w[n,j], ",  mr_w[n+1,j] = ", mr_w[n+1,j],
            #     "\nt_c[n, j] = ", t_c[n,j], ", t_c[n+1,j] = ", t_c[n+1,j],
            #     "\ndtcdz_param = ", dtcdz_param
            #     )
            ## END DEBUGGING ####################################################

        if n<s_len: w_c[n+1,j]=np.nan

    return (w_c, mse_c, mr_w, t_c, B, mflux, entr, detr, t_va, t_vc, mr_i, \
            mr_va, mr_vc, mr_cond, mr_auto, entrT_list)
