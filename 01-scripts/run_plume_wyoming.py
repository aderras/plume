import single_plume_model as spm


from datetime import datetime
from siphon.simplewebservice.wyoming import WyomingUpperAir
from metpy.units import units
import metpy.calc as mpcalc
import numpy as np

if __name__ == "__main__":

    """Choose one of the following dates and stations to acquire data."""
    # date = datetime(2017,9,10,0)
    # station = '72201' # Key West: 72201 or 'EYW' CAPE: 360-440

    # date = datetime(2022,5,30,12)
    # station = '82332' # Manaus (Aeroporto), Brazil CAPE:950-1100

    date = datetime(2022,5,30,12)
    station = '72206' # Jacksonville, FL CAPE:2200-2400

    df = WyomingUpperAir.request_data(date,station)

    P  = df['pressure'].values * units(df.units['pressure'])
    T  = df['temperature'].values * units(df.units['temperature'])
    Z  = df['height'].values * units(df.units['height'])
    Td = df['dewpoint'].values * units(df.units['dewpoint'])
    SH = mpcalc.specific_humidity_from_dewpoint(P,Td)

    P2 = np.array(P)
    T2 = np.array(T) + 273.15 # Convert to Kelvin
    Z2 = np.array(Z)
    SH2 = SH
    SPM_Z = np.arange(0, 20, .25)

    # Main lines to run the single plume model

    # 1. Prep the sounding
    sounding = spm.prep_sounding(Z2, P2, T2, SH2)

    sol = spm.run_single_plume(sounding, assume_entr=True)
    spm.save_as_csv(sol)
