import single_plume_model as spm
import helpers

if __name__ == "__main__":

    ellingson_df = helpers.import_ellingson_sounding()
    sounding = spm.prep_sounding(ellingson_df.z, ellingson_df.p, ellingson_df.t,
                                ellingson_df.sh)

    sol = spm.run_single_plume(sounding, assume_entr=True)
    spm.save_as_csv(sol)
