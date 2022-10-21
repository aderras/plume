import single_plume_model as spm
import helpers
import weighting_functions as wf

if __name__ == "__main__":

    import plume_functions

    ellingson_df = helpers.import_ellingson_sounding()
    sounding = spm.prep_sounding(ellingson_df.z, ellingson_df.p, ellingson_df.t,
                                ellingson_df.sh)

    sol = spm.run_single_plume(sounding, assume_entr=True)
    helpers.save_dict_elems_as_csv(sol)
    helpers.save_dict_elems_as_csv(sounding)

    sol_weighted = wf.get_weighted_profile(sol, sounding, cth=10.0)

    helpers.save_dict_elems_as_csv(sol_weighted, suffix="_weighted")
