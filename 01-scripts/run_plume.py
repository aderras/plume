import single_plume_model as spm
import helpers
import weighting_functions as wf
import constants

if __name__ == "__main__":

    import plume_functions

    ellingson_df = helpers.import_ellingson_sounding()
    sounding = spm.prep_sounding(ellingson_df.z, ellingson_df.p, ellingson_df.t,
                                ellingson_df.sh)

    storage = spm.initialize_storage(len(sounding[0]),len(constants.entrT_list))

    sol = spm.run_single_plume(storage, sounding,  assume_entr=True)
    helpers.save_vec_elems_as_csv(sounding[:7], ["z","p","t","sh","rho"])
    helpers.save_vec_elems_as_csv(sol,
            ["w_c", "mse_c", "q_w", "t_c", "B", "mflux", "entr", "detr", "t_va",\
            "t_vc", "q_i", "q_va", "q_vc", "entrT_list"])

    sol_weighted = wf.get_weighted_profile(sol, sounding, cth=10.0)
    helpers.save_dict_elems_as_csv(sol_weighted, suffix="_weighted")
