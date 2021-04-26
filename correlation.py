import numpy as np
import netCDF4 as nc
from matplotlib import pyplot as plt
import os, sys
from MakeRGB import my_cmap
from copy import deepcopy
from scipy import signal


def pearson(image1: np.array, image2: np.array, rad: int):
    mask1 = np.isnan(image1)
    image1[mask1] = np.nanmedian(image1)
    mask2 = np.isnan(image2)
    image2[mask2] = np.nanmedian(image2)
    kernel = np.ones((rad, rad)) / rad / rad
    image1_blurred = signal.convolve2d(image1, kernel, boundary='symm', mode='same')
    image2_blurred = signal.convolve2d(image2, kernel, boundary='symm', mode='same')
    image1_square = (image1 - image1_blurred) ** 2
    image2_square = (image2 - image2_blurred) ** 2
    image1_square_blurred = signal.convolve2d(image1_square, kernel, boundary='symm', mode='same')
    image2_square_blurred = signal.convolve2d(image2_square, kernel, boundary='symm', mode='same')
    image1_std = np.sqrt(image1_square_blurred)
    image2_std = np.sqrt(image2_square_blurred)

    product = (image1 - image1_blurred) * (image2 - image2_blurred)
    product_blurred = signal.convolve2d(product, kernel, boundary='symm', mode='same')
    correlation = product_blurred / image1_std / image2_std
    correlation[mask1] = np.NaN
    # print((np.min(image1_square_blurred)))
    return correlation


map_cloud = deepcopy(my_cmap)
map_cloud.set_over((128 / 255.0, 128 / 255.0, 128 / 255.0))
path = r"C:\Users\Alexander Semenov\Desktop\ABI_ACSPO_CODE\cloudmasktest\tiles_pics\all_areas"
output_figures_path = r"D:\Users\Alexander\ACSPO\correlation_map"
labelsize = 8
fontsize = 8
files = os.listdir(path)
# files = files[0:3]
for file in files:
    path_tile = os.path.join(path, file)
    nc_tile = nc.Dataset(path_tile, "r")
    sst_regression = np.array(nc_tile.variables["sst_regression"][:])
    sst_reynolds = np.array(nc_tile.variables["sst_reynolds"][:])
    validation_mask = np.array(nc_tile.variables["validation_mask"][:])
    original_mask = np.array(nc_tile.variables["original_mask"][:])
    atm_diff = np.array(nc_tile.variables["atm_diff"][:])
    brightness_temp_ch7 = np.array(nc_tile["brightness_temp_ch7"][:])  # 4nm
    brightness_temp_ch14 = np.array(nc_tile["brightness_temp_ch14"][:])  # 4nm
    brightness_temp_crtm_ch7 = np.array(nc_tile["brightness_temp_crtm_ch7"][:])  # 4nm
    brightness_temp_crtm_ch14 = np.array(nc_tile["brightness_temp_crtm_ch14"][:])  # 4nm
    brightness_temp_ch11 = np.array(nc_tile["brightness_temp_ch11"][:])  # 4nm
    brightness_temp_ch15 = np.array(nc_tile["brightness_temp_ch15"][:])  # 4nm
    brightness_temp_crtm_ch11 = np.array(nc_tile["brightness_temp_crtm_ch11"][:])  # 4nm
    brightness_temp_crtm_ch15 = np.array(nc_tile["brightness_temp_crtm_ch15"][:])  # 4nm
    dt = sst_regression - sst_reynolds
    dt_original = deepcopy(dt)
    first_var = deepcopy(sst_regression-atm_diff)
    second_var = deepcopy(atm_diff)
    correlation = pearson(first_var, second_var, 7)

    fig = plt.figure(figsize=(15, 10))
    fig.add_subplot(1, 4, 1)
    plt.imshow(correlation, interpolation="none", cmap=my_cmap, vmin=-1, vmax=1)
    cbar3 = plt.colorbar(fraction=0.046, pad=0.04)
    cbar3.ax.tick_params(labelsize=labelsize)
    plt.xticks([])
    plt.yticks([])
    plt.title("Correlation coefficient \n between sst-atm_diff and atm_diff ,\n Window_size = 7", fontsize=10)
    fig.add_subplot(1, 4, 2)
    plt.imshow(dt, interpolation="none", cmap=my_cmap, vmin=-2, vmax=2)
    cbar3 = plt.colorbar(fraction=0.046, pad=0.04)
    cbar3.ax.tick_params(labelsize=labelsize)
    plt.xticks([])
    plt.yticks([])
    plt.title("dt_analysis, all sky, K", fontsize=fontsize)
    fig.add_subplot(1, 4, 3)
    dt[dt > 2] = 1.9995
    dt[validation_mask > 0] = 4.0
    plt.imshow(dt, interpolation="none", cmap=map_cloud, vmin=-2, vmax=2)
    cbar3 = plt.colorbar(fraction=0.046, pad=0.04)
    cbar3.ax.tick_params(labelsize=labelsize)
    plt.xticks([])
    plt.yticks([])
    plt.title("dt_analysis, with validation cloud mask, K", fontsize=fontsize)

    fig.add_subplot(1, 4, 4)
    dt_original[dt_original > 2] = 1.9995
    dt_original[original_mask > 0] = 4.0
    plt.imshow(dt_original, interpolation="none", cmap=map_cloud, vmin=-2, vmax=2)
    cbar3 = plt.colorbar(fraction=0.046, pad=0.04)
    cbar3.ax.tick_params(labelsize=labelsize)
    plt.xticks([])
    plt.yticks([])
    plt.title("dt_analysis, with original cloud mask, K", fontsize=fontsize)

    plt.savefig(os.path.join(output_figures_path, file + ".jpg"))
    plt.close(fig)
sys.exit()
