import numpy as np
import netCDF4 as nc
from matplotlib import pyplot as plt
import os, sys
from MakeRGB import my_cmap

dict_BTS_9 = {}
dict_BTS_7 = {}
dict_BTS_10 = {}

def get_BTS():
    path_7 = r"D:\Users\Alexander\ACSPO\Opencv\channels_7"
    path_9 = r"D:\Users\Alexander\ACSPO\Opencv\channels_9"
    path_10 = r"D:\Users\Alexander\ACSPO\Opencv\channels_10"
    files = os.listdir(path_9)

    for file in files:
        nc_data = nc.Dataset(os.path.join(path_9, file), "r")

        Rad = np.array(nc_data.variables["Rad"][:])
        DQF = np.array(nc_data.variables["DQF"][:])

        plank_fk1 = nc_data.variables["planck_fk1"]
        plank_fk2 = nc_data.variables["planck_fk2"]

        plank_bc1 = nc_data.variables["planck_bc1"]
        plank_bc2 = nc_data.variables["planck_bc2"]

        # print(nc_data.variables.keys())

        BTs = (plank_fk2 / (np.log(plank_fk1 / Rad) + 1.0) - plank_bc1) / plank_bc2

        BTs[DQF != 0] = np.NaN
        dict_BTS_9[file[:15]] = BTs

    files = os.listdir(path_10)

    for file in files:
        nc_data = nc.Dataset(os.path.join(path_10, file), "r")

        Rad = np.array(nc_data.variables["Rad"][:])
        DQF = np.array(nc_data.variables["DQF"][:])

        plank_fk1 = nc_data.variables["planck_fk1"]
        plank_fk2 = nc_data.variables["planck_fk2"]

        plank_bc1 = nc_data.variables["planck_bc1"]
        plank_bc2 = nc_data.variables["planck_bc2"]

        # print(nc_data.variables.keys())

        BTs = (plank_fk2 / (np.log(plank_fk1 / Rad) + 1.0) - plank_bc1) / plank_bc2

        BTs[DQF != 0] = np.NaN
        dict_BTS_10[file[:15]] = BTs

    files = os.listdir(path_7)

    for file in files:
        nc_data = nc.Dataset(os.path.join(path_7, file), "r")

        Rad = np.array(nc_data.variables["Rad"][:])
        DQF = np.array(nc_data.variables["DQF"][:])

        plank_fk1 = np.array(nc_data.variables["planck_fk1"][:])
        plank_fk2 = np.array(nc_data.variables["planck_fk2"][:])

        plank_bc1 = np.array(nc_data.variables["planck_bc1"][:])
        plank_bc2 = np.array(nc_data.variables["planck_bc2"][:])



        # print(nc_data.variables.keys())

        BTs = (plank_fk2 / (np.log(plank_fk1 / Rad) + 1.0) - plank_bc1) / plank_bc2
            # BTs *  plank_bc2 + plank_bc1 = plank_fk2 / (np.log(plank_fk1 / Rad) + 1.0)
        # np.log(plank_fk1 / Rad) + 1.0 = plank_fk2 / (BTs *  plank_bc2 + plank_bc1)
        # np.log(plank_fk1 / Rad) = plank_fk2 / (BTs *  plank_bc2 + plank_bc1) - 1.0
        # (plank_fk1 / Rad)  = np.exp(plank_fk2 / (BTs *  plank_bc2 + plank_bc1) - 1.0)
        # Rad = plank_fk1 /np.exp(plank_fk2 / (BTs *  plank_bc2 + plank_bc1) - 1.0)
        BTs[DQF != 0] = np.NaN
        dict_BTS_7[file[:15]] = BTs
# string = "Val_ACSPO_V2.80_G16_ABI_2020-05-17_0700-0710_20200914.183501_R04570-C03487.nc"
#
#
# sys.exit()
# get_BTS()