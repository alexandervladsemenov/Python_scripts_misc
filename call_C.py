import numpy as np
import netCDF4 as nc
from matplotlib import pyplot as plt
from matplotlib import cm
import os
import sys
import copy
from my_map import my_cmap

path_to_dsst_inp: str = r"C:\Users\Alexander Semenov\Desktop\ABI_ACSPO_CODE\OpenCV_GRANULES\dsst"
path_to_atm_inp: str = r"C:\Users\Alexander Semenov\Desktop\ABI_ACSPO_CODE\OpenCV_GRANULES\atm_layers"
path_to_tiles: str = r"C:\Users\Alexander Semenov\Desktop\ABI_ACSPO_CODE\cloudmasktest\tiles_pics\all_areas"
path_out_tiles_binaries = r"C:\Users\Alexander Semenov\Desktop\ABI_ACSPO_CODE\cloudmasktest\tiles_pics\binaries_true_dsst"
path_check_file = r"C:\Users\Alexander Semenov\Desktop\ABI_ACSPO_CODE\cloudmasktest\tiles_pics\dsst_dt_analysis"

files_dsst_full = os.listdir(path_to_dsst_inp)
files_atm_full = os.listdir(path_to_atm_inp)
# ACSPO_V2.80B04_G16_ABI_2018-06os.listdir(path_to_dsst_inp)-27_0700-0715_20210225.133518.nc.bin

# time_date_full = files_dsst_full[0][23:43]  # full_time_date

all_dsst_fulls: dict = {}
all_atm_fulls = {}
all_big_granules : dict = {}

file_with_mask = r"C:\Users\Alexander Semenov\Desktop\ABI_ACSPO_CODE\OpenCV_GRANULES\masks\ACSPO_V2.80B04_G16_ABI_2020-07-21_0700-0710_20210219.191551.nc"


big_granules = r"C:\Users\Alexander Semenov\Desktop\ABI_ACSPO_CODE\OpenCV_GRANULES\masks"
mask_nc = nc.Dataset(file_with_mask,'r')

sst_regression = np.array(mask_nc.variables["sst_regression"][:])

mask_full = np.isnan(sst_regression)



for file_full in files_dsst_full:
    dsst = np.fromfile(os.path.join(path_to_dsst_inp, file_full), dtype=np.float32)
    time_date_full: str = file_full[23:43]
    dsst = dsst.reshape((5424, 5424))
    dsst[mask_full] = np.NaN
    all_dsst_fulls[time_date_full] = dsst

for file_full in files_atm_full:
    atm = np.fromfile(os.path.join(path_to_atm_inp, file_full), dtype=np.float32)
    time_date_full: str = file_full[23:43]
    atm = atm.reshape((5424, 5424))
    all_atm_fulls[time_date_full] = atm

files_dsst_tiles = os.listdir(path_to_tiles)

for file_tiles in files_dsst_tiles:
    print(file_tiles[24:44])  # date_time
    print(int(file_tiles[62:67]))  # row
    print(int(file_tiles[69:74]))  # cols
    time_date = file_tiles[24:44]
    r = int(file_tiles[62:67])
    c = int(file_tiles[69:74])
    name = os.path.join(path_to_tiles, file_tiles)
    data = nc.Dataset(name, "r")

    sst_reg_tile = np.array(data.variables["sst_regression"][:])
    sst_rey_tile = np.array(data.variables["sst_reynolds"][:])

    original_mask = np.array(data.variables["original_mask"][:])
    validation_mask = np.array(data.variables["validation_mask"][:])

    dr, dc = sst_reg_tile.shape
    dsst_full = all_dsst_fulls[time_date]
    atm_full = all_atm_fulls[time_date]
    dsst_to_out = dsst_full[r:r + dr, c:c + dc]
    atm_out = atm_full[r:r + dr, c:c + dc]

    dsst_to_out_broad = dsst_full[r - dr:r + dr * 2, c - dc:c + dc * 2]

    dsst_to_out[np.isnan(sst_reg_tile)] = np.NaN
    dsst_to_out[np.isnan(sst_rey_tile)] = np.NaN

    low = np.nanmin(sst_rey_tile) - 2.0
    high = np.nanmax(sst_reg_tile)

    file_name_bin = file_tiles + ".bin"
    dsst_to_out.astype('float32').tofile(os.path.join(path_out_tiles_binaries, file_name_bin))
    fig = plt.figure(figsize=(20, 10))

    fig.add_subplot(2, 3, 1)
    plt.imshow(dsst_to_out_broad, interpolation="none", cmap=my_cmap, vmin=-1.8, vmax=1.8)

    fig.add_subplot(2, 3, 2)
    plt.imshow(sst_reg_tile - sst_rey_tile, interpolation="none", cmap=my_cmap, vmin=-1.8, vmax=1.8)
    cbar4 = plt.colorbar(fraction=0.046, pad=0.04)

    cbar4.ax.tick_params(labelsize=30)
    fig.add_subplot(2, 3, 3)
    plt.imshow(sst_reg_tile, interpolation="none", cmap=my_cmap, vmin=low, vmax=high)

    cbar4 = plt.colorbar(fraction=0.046, pad=0.04)
    cbar4.ax.tick_params(labelsize=30)

    # fig.add_subplot(2, 4, 5)
    # plt.imshow( atm_out, interpolation="none", cmap=my_cmap, vmin=np.nanmin(atm_out), vmax=np.nanmax(atm_out))
    # cbar5 = plt.colorbar(fraction=0.046, pad=0.04)
    # cbar5.ax.tick_params(labelsize=30)
    #
    #
    fig.add_subplot(2, 3, 4)
    plt.imshow(sst_rey_tile, interpolation="none", cmap=my_cmap, vmin=low, vmax=high)

    new_cp = copy.deepcopy(my_cmap)
    new_cp.set_over("gray")

    fig.add_subplot(2, 3, 5)
    dt_or_nmask = np.where(original_mask, high + 100, sst_reg_tile)

    plt.imshow(dt_or_nmask, interpolation="none", cmap=new_cp, vmin=low, vmax=high)

    fig.add_subplot(2, 3, 6)
    dt_val_nmask = np.where(validation_mask, high + 100, sst_reg_tile)

    plt.imshow(dt_val_nmask, interpolation="none", cmap=new_cp, vmin=low, vmax=high)
    cbar6 = plt.colorbar(fraction=0.046, pad=0.04)
    cbar6.ax.tick_params(labelsize=30)

    #
    # fig.add_subplot(2, 4, 8)
    # new_cp2 = copy.deepcopy(new_cp)
    # new_cp2.set_under("pink")
    # mask = (atm_out-sst_rey_tile>-8)& (sst_rey_tile<sst_reg_tile)
    # plt.imshow(mask.astype("float32"), interpolation="none", cmap=new_cp2, vmin=0, vmax=1)
    #
    # cbar6= plt.colorbar(fraction=0.046, pad=0.04)
    # cbar6.ax.tick_params(labelsize=30)

    name_fig_save = "comp_" + file_tiles + ".png"
    plt.savefig(os.path.join(path_check_file, name_fig_save))
sys.exit()
