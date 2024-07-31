# Installation instructions

The following instructions detail installation of libraries for this program on Fedora 36.

**Required packages:** numpy scipy metpy numba pickle pandas siphon

To install required packages in a new conda environment, called "plume" below:

  ```
  conda config --add channels conda-forge
  conda create -n plume python=3.9 numpy scipy numba pandas jupyterlab cdsapi xarray rioxarray skimage
  pip install siphon
  pip install metpy
  ```

# Running the program

To test the program on the Ellingson sounding profile, saved locally, run `python run_plume_ellingson.py` from the "01-scripts/" folder. Plot the results of this file by running `python plot_results_ellingson.py`.

Alternatively, you can run the program on data retrieved from Wyoming Weather Lab soundings in the script `run_plume_wyoming.py`. This script uses the WyomingUpperAir module to download data from the server. You can plot the results by running `python plot_results_wyoming.py`.
