# Installation instructions

The following instructions detail installation of libraries for this program on Fedora 36.

**Required packages:** numpy scipy metpy numba pickle pandas siphon

To install required packages in a new conda environment:
  `conda config --add channels confa-forge`
  `conda create -n plume python=3.9 numpy scipy numba pandas jupyterlab`
  `pip install siphon`
  `pip install metpy`

# Running the program

To test the program with local data, run `python run_plume.py` from the "01-scripts/" folder. To run the program with data downloaded from MetPy, run `python run_plume_wyoming.py` from the same folder.
