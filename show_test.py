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

nc_file_path = "C:/Users/Alexander Semenov/Desktop/ABI_ACSPO_CODE/cloudmasktest/tiles_pics/all_areas/Val_ACSPO_V2.80_G16_ABI_2020-05-17_0700-0710_20200914.183501_R04611-C03332.nc"

nc_file_path_nan = "C:/Users/Alexander Semenov/Desktop/ABI_ACSPO_CODE/cloudmasktest/tiles_pics/all_areas/Val_ACSPO_V2.80_G16_ABI_2020-05-17_0700-0710_20200914.183501_R04375-C03410.nc"

# my_cmap.set_over("gray")
my_cmap.set_bad("saddlebrown")

f1 = nc.Dataset(nc_file_path, "r")
f2 = nc.Dataset(nc_file_path_nan, "r")

sst_reg_tile1 = np.array(f1.variables["sst_regression"][:])
sst_rey_tile1 = np.array(f1.variables["sst_reynolds"][:])
validation_mask1 = np.array(f1.variables["validation_mask"][:])
original_mask1 = np.array(f1.variables["original_mask"][:])
individual1 = np.array(f1.variables["individual_clear_sky_tests_results"][:],dtype=np.uint8)
extra1 = np.array(f1.variables["extra_byte_clear_sky_tests_results"][:])




sst_reg_tile2 = np.array(f2.variables["sst_regression"][:], dtype=np.float32)
sst_rey_tile2 = np.array(f2.variables["sst_reynolds"][:], dtype=np.float32)
validation_mask2 = np.array(f2.variables["validation_mask"][:])
original_mask2 = np.array(f2.variables["original_mask"][:])
individual2 = np.array(f2.variables["individual_clear_sky_tests_results"][:],dtype=np.uint8)
extra2 = np.array(f2.variables["extra_byte_clear_sky_tests_results"][:])

switch = True
if switch:
    dt = paint(sst_reg_tile1 - sst_rey_tile1)
    sst_reg_tile = paint(sst_reg_tile1)
    sst_rey_tile = paint(sst_rey_tile1)
    sst_rey_tile_copy = copy.deepcopy(sst_rey_tile1)
    nan_mask = np.isnan(sst_reg_tile1)
    original_mask = original_mask1
    extra = copy.deepcopy(extra1)
    individual = copy.deepcopy(individual1)
    to_plot = copy.deepcopy(sst_reg_tile1)
    validation_mask= validation_mask1
else:
    dt = paint(sst_reg_tile2 - sst_rey_tile2)
    sst_reg_tile = paint(sst_reg_tile2)
    sst_rey_tile = paint(sst_rey_tile2)
    sst_rey_tile_copy = copy.deepcopy(sst_rey_tile2)
    nan_mask = np.isnan(sst_reg_tile2)
    original_mask = original_mask2
    extra = copy.deepcopy(extra2)
    individual = copy.deepcopy(individual2)
    to_plot = copy.deepcopy(sst_reg_tile2)
    validation_mask= validation_mask2
# the same for all tiles
# defining area not to erode
sst_reg_tile_blur = cv2.GaussianBlur(sst_reg_tile, ksize=(3, 3), sigmaX=0)

laplace_sst_reg_tile_blur = cv2.Laplacian(sst_reg_tile_blur, cv2.CV_32FC1, ksize=3)

erode_low_limit = 2

not_erode_mask = (laplace_sst_reg_tile_blur > erode_low_limit)

radius_of_erode = 5

not_erode_mask = (cv2.GaussianBlur(not_erode_mask.astype("float"), ksize=(radius_of_erode, radius_of_erode),
                                   sigmaX=0) > 0) & (dt < -0.5)

# computing the gradient
# Define kernel for x differences
kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
# Define kernel for y differences
ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

# Perform x convolution
x = ndimage.convolve(sst_rey_tile, kx)
# Perform y convolution
y = ndimage.convolve(sst_rey_tile, ky)
sobel = np.hypot(x, y)
grad_low = 2

suspect_mask = sobel > grad_low

rad_broaden = 5
suspect_mask = (cv2.GaussianBlur(suspect_mask.astype("float"), ksize=(rad_broaden, rad_broaden), sigmaX=0) > 0)
# eroding the reference

erosion_size = 20

element = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                    (erosion_size, erosion_size))

eroded_sst_ref = cv2.erode(sst_rey_tile, element)

# preparing dsst

where_to_erode = suspect_mask & ~ not_erode_mask

dsst = np.where(where_to_erode, sst_reg_tile - eroded_sst_ref, dt)

# plt.imshow(np.concatenate((dsst,dt,),axis=0), interpolation="none", cmap=my_cmap,vmin=-1.8,vmax=1.8)
#
# plt.colorbar()
# plt.show()

# Calc the mask


# blurring

dsst_blurred = cv2.GaussianBlur(dsst, ksize=(3, 3), sigmaX=0)

dsst_blurred[dsst_blurred < -1.8] = -1.8
dsst_blurred[dsst_blurred > 1.8] = 1.8

dsst_image = np.array((dsst_blurred[~nan_mask] + 1.8) / 3.6 * 255, dtype=np.uint8)

threshold, mask = cv2.threshold(dsst_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU);

I_pixel = threshold

I_threshold = I_pixel / 225.0 * 3.6 - 1.8

print(I_threshold)

static_mask = individual >> 2 & 1
adaptive_mask = (((individual >> 3 & 1) != static_mask) == 1) & ~where_to_erode
rgct_mask = individual >> 4 & 1
uniformity_mask = individual >> 6 & 1
cross_corr_mask = individual >> 7 & 1
pos_mask = extra >> 3 & 1
normal_mask = static_mask | adaptive_mask | rgct_mask | uniformity_mask | cross_corr_mask | pos_mask
non_mask = (dsst < -1.8)   | (adaptive_mask == 1) |  (rgct_mask==1) | (uniformity_mask==1) | (cross_corr_mask==1) | (pos_mask==1)

mask_static = dsst_blurred < -.45
# mask_static = dt<I_threshold

laplace_dsst_tile_blur = cv2.Laplacian(cv2.GaussianBlur(dsst, ksize=(3, 3), sigmaX=0), cv2.CV_32FC1, ksize=5)

mask_laplace = (np.abs(laplace_dsst_tile_blur) > 0.5) & (dsst_blurred < -0.4)

total_mask =  non_mask
print(I_threshold)

my_cmap.set_over("gray")

dt[dt > 1.8] = 1.8

dt_pure = copy.deepcopy(dt)

dt[total_mask] = 2.0
dt[nan_mask] = np.NaN

vmax = np.nanmax(to_plot)

T_min = np.nanmin(sst_rey_tile_copy)
T_max = np.nanmax(sst_rey_tile_copy)
print(T_max,T_min)
T_max_lim = T_max + 2
T_min_lim = T_min  -2
T_fill_over = T_max_lim + 5
T_fill_under = T_min_lim - 5


to_plot_or = np.where(original_mask, T_fill_over, to_plot)
to_plot[total_mask] = T_fill_over
to_plot[nan_mask] = np.NaN

dsst[dsst > 1.8] = 1.8
dsst_pure = copy.deepcopy(dsst)
dsst_pure_copy = copy.deepcopy(dsst)

dsst[total_mask] = 2.0
dsst[nan_mask] = np.NaN

fig = plt.figure(figsize=(20, 10))

fig.add_subplot(1, 2, 1)

plt.imshow(to_plot, interpolation="none", cmap=my_cmap, vmin=T_min_lim, vmax=T_max_lim)
cbar = plt.colorbar(fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=25)
plt.xticks([])
plt.yticks([])

fig.add_subplot(1, 2, 2)
plt.imshow(to_plot_or, interpolation="none", cmap=my_cmap, vmin=T_min_lim, vmax=T_max_lim)
cbar = plt.colorbar(fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=25)
plt.xticks([])
plt.yticks([])
if switch:
    plt.savefig(os.path.join( "images/to_present_{}.png".format("figure_5")))
else:
    plt.savefig(os.path.join( "images/to_present_{}.png".format("figure_6")))


    plt.close(fig)
# plt.imshow(dt, interpolation="none", cmap=my_cmap, vmin=-1.8, vmax=1.8)
# plt.colorbar()
# plt.show()
#
# plt.imshow(dt_pure, interpolation="none", cmap=my_cmap, vmin=-1.8, vmax=1.8)
# plt.colorbar()
# plt.show()

sys.exit()
