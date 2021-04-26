import cv2
import os
import numpy as np
from MakeRGB import my_cmap
from matplotlib import pyplot as plt
from copy import deepcopy
import sys
import netCDF4 as nc
from bt_tests import save_obj, load_obj
thresh = 2.0
dilatation_size = 11
element = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                    (dilatation_size, dilatation_size))

map_cloud = deepcopy(my_cmap)
map_cloud.set_over((128 / 255.0, 128 / 255.0, 128 / 255.0))

full_out_path = r"D:\Users\Alexander\ACSPO\Opencv\btd_acspo"

files_to_delete = os.listdir(full_out_path)

for file_d in files_to_delete:
    os.remove(os.path.join(full_out_path, file_d))


def to_grayscale(array: np.array, dt_max=2, dt_min=-2):
    median = np.nanmedian(array)
    max_val = np.nanmax(array)
    min_val = np.nanmin(array)
    print(min_val, max_val, median)
    array[np.isnan(array)] = median
    max_val = min(dt_max, max_val)
    min_val = max(dt_min, min_val)
    array[array > max_val] = max_val
    array[array < min_val] = min_val
    result: np.array = (array - min_val) / (max_val - min_val) * 255
    return result.astype(np.uint8)


path = r"C:\Users\Alexander Semenov\Desktop\ABI_ACSPO_CODE\cloudmasktest\tiles_pics\all_areas"

files = os.listdir(path)
files.sort()
param = 1.8

rad = 15
tests_data = load_obj("data")
n_figures = 1
n_col = 6

label_size = 15
font_size = 15

index = 6

show_other = False
if ~show_other:
    n_col = 3

for index in range(len(files)):
    file = files[index]

    nc_data = nc.Dataset(os.path.join(path, file), "r")

    sst_reynolds = np.array(nc_data.variables["sst_reynolds"])
    sst_regression = np.array(nc_data.variables["sst_regression"])
    individual = np.array(nc_data["individual_clear_sky_tests_results"][:])
    nc_data.close()
    adaptive_mask = individual >> 3 & 1
    gradientx = cv2.Sobel(sst_reynolds, ddepth=cv2.CV_32FC1, dx=1, dy=0)

    gradienty = cv2.Sobel(sst_reynolds, ddepth=cv2.CV_32FC1, dx=0, dy=1)

    abs_gradient = gradientx ** 2 + gradienty ** 2
    mask_gradient = abs_gradient > param
    mask_gradient = cv2.GaussianBlur(mask_gradient.astype(np.float32), (rad, rad), 0)

    sst_reynolds_max = cv2.dilate(sst_reynolds, element)
    sst_reynolds_min = cv2.erode(sst_reynolds, element)

    corrected_mask = ~(mask_gradient > 0) & (adaptive_mask > 0)
    corrected_mask = ~((sst_reynolds_max - sst_reynolds_min) > thresh) & (adaptive_mask > 0)

    mask_BTD = tests_data["FULL_BTD_MASK"][index, :, :]

    delta_sst = tests_data["delta_sst"][index, :, :]
    total_mask = tests_data["Individual"][index, :, :]
    delta_sst[delta_sst > 2.0] = 2.0
    delta_sst[delta_sst < -2.0] = -2.0
    Validation = tests_data["Validation"][index, :, :]
    Original = tests_data["Original"][index, :, :]
    nan_mask = np.isnan(tests_data["delta_sst"][index, :, :])
    fig = plt.figure(figsize=(20, 10))
    fig.add_subplot(n_figures, n_col, 1)
    to_show = np.where((mask_BTD > 0) | corrected_mask, 100.0, delta_sst)
    to_show[nan_mask] = np.NaN
    plt.imshow(to_show, interpolation="none", vmin=-2, vmax=2, cmap=map_cloud)
    plt.title("BTD_MASK + Static_Adaptive\ncorrected by gradient", fontsize=font_size)
    cbar3 = plt.colorbar(fraction=0.046, pad=0.04)
    cbar3.ax.tick_params(labelsize=label_size)
    plt.xticks([])
    plt.yticks([])
    # plotting the Validation mask
    fig.add_subplot(n_figures, n_col, 2)
    to_show = np.where(Validation > 0, 100.0, delta_sst)
    to_show[nan_mask] = np.NaN
    plt.imshow(to_show, interpolation="none", vmin=-2, vmax=2, cmap=map_cloud)
    plt.title("Validation", fontsize=font_size)
    cbar3 = plt.colorbar(fraction=0.046, pad=0.04)
    cbar3.ax.tick_params(labelsize=label_size)
    plt.xticks([])
    plt.yticks([])

    fig.add_subplot(n_figures, n_col, 3)
    to_show = np.where((mask_BTD > 0), 100.0, delta_sst)
    to_show[nan_mask] = np.NaN
    plt.imshow(to_show, interpolation="none", vmin=-2, vmax=2, cmap=map_cloud)
    plt.title("BTD_ORIGINAL", fontsize=font_size)
    cbar3 = plt.colorbar(fraction=0.046, pad=0.04)
    cbar3.ax.tick_params(labelsize=label_size)
    plt.xticks([])
    plt.yticks([])

    if show_other:




        fig.add_subplot(n_figures, n_col, 4)
        to_show = np.where((Original > 0), 100.0, delta_sst)
        to_show[nan_mask] = np.NaN
        plt.imshow(to_show, interpolation="none", vmin=-2, vmax=2, cmap=map_cloud)
        plt.title("ACSPO ORIGINAL", fontsize=font_size)
        cbar3 = plt.colorbar(fraction=0.046, pad=0.04)
        cbar3.ax.tick_params(labelsize=label_size)
        plt.xticks([])
        plt.yticks([])

        fig.add_subplot(n_figures, n_col, 5)
        to_show = sst_reynolds
        to_show[nan_mask] = np.NaN
        plt.imshow(to_show, interpolation="none",cmap=my_cmap)
        plt.title("sst_reynolds", fontsize=font_size)
        cbar3 = plt.colorbar(fraction=0.046, pad=0.04)
        cbar3.ax.tick_params(labelsize=label_size)
        plt.xticks([])
        plt.yticks([])

        fig.add_subplot(n_figures, n_col, 6)
        to_show = sst_regression
        to_show[nan_mask] = np.NaN
        plt.imshow(to_show, interpolation="none",cmap=my_cmap,vmin=np.nanmin(sst_reynolds)-2.0)
        plt.title("sst_regression", fontsize=font_size)
        cbar3 = plt.colorbar(fraction=0.046, pad=0.04)
        cbar3.ax.tick_params(labelsize=label_size)
        plt.xticks([])
        plt.yticks([])

    plt.savefig(os.path.join(full_out_path, file + ".jpg"))
    plt.close()

sys.exit()
