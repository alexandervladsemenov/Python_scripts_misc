import numpy as np
import netCDF4 as nc
from matplotlib import pyplot as plt
from matplotlib import cm
import os
import sys
import copy
import cv2
from scipy import ndimage
from my_map import my_cmap, paint

nc_folder = "C:/Users/Alexander Semenov/Desktop/ABI_ACSPO_CODE/cloudmasktest/tiles_pics/all_areas/"

present_folder = r"C:\Users\Alexander Semenov\Desktop\ABI_ACSPO_CODE\cloudmasktest\tiles_pics\to_present"

warm_map = copy.deepcopy(my_cmap)
warm_map.set_under("pink")
warm_map.set_over("gray")

files = os.listdir(nc_folder)

for file in files:
    name = os.path.join(nc_folder, file)

    data = nc.Dataset(name, "r")

    keys = data.variables.keys()

    sst_regression = np.array(data.variables["sst_regression"][:])
    sst_reynolds = np.array(data.variables["sst_reynolds"][:])
    individual = np.array(data.variables["individual_clear_sky_tests_results"][:])
    extra = np.array(data.variables["extra_byte_clear_sky_tests_results"][:])
    original_mask = np.array(data.variables["original_mask"][:])
    validation_mask = np.array(data.variables["validation_mask"][:])
    static_mask = individual >> 2 & 1
    adaptive_mask = individual >> 3 & 1
    rgct_mask = individual >> 4 & 1
    uniformity_mask = individual >> 6 & 1
    cross_corr_mask = individual >> 7 & 1
    pos_mask = extra >> 3 & 1
    normal_mask = static_mask | adaptive_mask | rgct_mask | uniformity_mask | cross_corr_mask | pos_mask
    warm_mask = (extra >> 0 & 1) | (extra >> 1 & 1)
    delta_sst = sst_regression - sst_reynolds
    fig = plt.figure(figsize=(20, 10))
    delta_max = 2
    delta_min = -2

    T_min = np.nanmin(sst_reynolds)
    T_max = np.nanmax(sst_reynolds)
    T_max_lim = T_max + delta_max
    T_min_lim = T_min + delta_min
    T_fill_over = T_max_lim + 5
    T_fill_under = T_min_lim - 5
    sst_regression_to_show = copy.deepcopy(sst_regression)
    sst_regression_to_show[sst_regression_to_show < T_min_lim] = T_min_lim
    sst_regression_to_show[sst_regression_to_show > T_max_lim] = T_max_lim
    sst_regression_to_show[normal_mask == 1] = T_fill_over
    sst_regression_old = copy.deepcopy(sst_regression_to_show)
    sst_regression_old[warm_mask == 1] = T_fill_over # T_fill_under

    fig.add_subplot(1, 2, 1)
    plt.imshow(sst_regression_to_show, interpolation="none", cmap=warm_map, vmin=T_min_lim,
               vmax=T_max_lim)  # need .AND. with mask_valid
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=25)
    plt.xticks([])
    plt.yticks([])

    fig.add_subplot(1, 2, 2)
    plt.imshow(sst_regression_old, interpolation="none", cmap=warm_map, vmin=T_min_lim,
               vmax=T_max_lim)  # need .AND. with mask_valid
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=25)
    plt.xticks([])
    plt.yticks([])

    # fig.add_subplot(1, 4, 3)
    # plt.imshow(delta_sst, interpolation="none", cmap=my_cmap, vmin=delta_min,
    #            vmax=delta_max)  # need .AND. with mask_valid
    # cbar = plt.colorbar(fraction=0.046, pad=0.04)
    # cbar.ax.tick_params(labelsize=25)
    # plt.xticks([])
    # plt.yticks([])
    #
    # fig.add_subplot(1, 4, 4)
    # plt.imshow(sst_reynolds, interpolation="none", cmap=my_cmap, vmin=T_min,
    #            vmax=T_max)  # need .AND. with mask_valid
    # cbar = plt.colorbar(fraction=0.046, pad=0.04)
    # cbar.ax.tick_params(labelsize=25)
    # plt.xticks([])
    # plt.yticks([])

    plt.savefig(os.path.join(present_folder, "no_warm_to_present_{}.png".format(file)))
    plt.close(fig)

    data.close()

sys.exit()
