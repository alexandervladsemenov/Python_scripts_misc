import os
from MakeRGB import my_cmap
from matplotlib import pyplot as plt
import cv2
import numpy as np
import netCDF4 as nc
from bt_tests import high_gradient_sst
from adaptive_test import adaptive_test
from copy import deepcopy


def get_BT_from_L1b(path: str):
    data = nc.Dataset(path, "r")
    Rad = np.array(data.variables["Rad"][:])
    DQF = np.array(data.variables["DQF"][:])

    plank_fk1 = data.variables["planck_fk1"]
    plank_fk2 = data.variables["planck_fk2"]

    plank_bc1 = data.variables["planck_bc1"]
    plank_bc2 = data.variables["planck_bc2"]

    BTs = (plank_fk2 / (np.log(plank_fk1 / Rad) + 1.0) - plank_bc1) / plank_bc2

    BTs[DQF != 0] = np.NaN
    return BTs


def get_Bt_from_L2P(path, name):
    data = nc.Dataset(path, mode="r")
    bt = np.array(data.variables[name][:])
    data.close()
    return bt


def draw_images(*ranges, **images):
    count = len(images)
    fig = plt.figure(figsize=(15 * count, 10))
    counter = 0
    font_size = 12
    label_size = 12
    i = 0
    for key in images:
        image = images[key]
        counter += 1
        fig.add_subplot(1, count, counter)
        val_min = ranges[i][0]
        val_max = ranges[i][1]
        color_map = ranges[i][2]
        i += 1
        plt.imshow(image, interpolation="none", vmin=val_min, vmax=val_max, cmap=color_map)
        plt.title(key, fontsize=font_size)
        cbar3 = plt.colorbar(fraction=0.046, pad=0.04)
        cbar3.ax.tick_params(labelsize=label_size)
        # plt.xticks([])
        # plt.yticks([])


def get_sst_cmc(path: str):
    data = nc.Dataset(path, mode="r")
    sst = np.array(data.variables["sst_regression"][:])
    cmc = np.array(data.variables["sst_reynolds"][:])
    data.close()
    return sst, cmc


def get_acspo_mask(path: str):
    data = nc.Dataset(path, mode="r")
    acspo_mask = np.array(data.variables["acspo_mask"][:]).astype(np.uint8)
    mask = ((acspo_mask >> 7 & 1) > 0) & ~((acspo_mask >> 6 & 1) > 0)
    data.close()
    return mask


def adaptive_boundaries_first_prototype(sst, cmc, threshold_min, threshold_max, threshold_gradient, radius, window_size,
                                        scale, cold_mask):
    delta_sst_original = sst - cmc
    grad_mask, cmc_min = high_gradient_sst(sst_reynolds=cmc, threshold=threshold_gradient, radius=radius)
    delta_sst_modified = sst - cmc_min
    mask_to_pass_for_adaptive = cold_mask & (delta_sst_original < threshold_min) & grad_mask
    adaptive_mask = adaptive_test(image=delta_sst_modified, window_size=window_size, scale=scale,
                                  mask=mask_to_pass_for_adaptive, threshold=threshold_min)
    # threshold_min = -1.5
    # threshold_max = -0.5
    return adaptive_mask & (delta_sst_original < threshold_max)


def adaptive_boundaries_second_prototype(sst, cmc, threshold_min, threshold_max, threshold_gradient, radius,
                                         window_size,
                                         scale, cold_mask):
    grad_mask, cmc_min = high_gradient_sst(sst_reynolds=cmc, threshold=threshold_gradient, radius=radius)
    delta_sst_modified = sst - cmc_min
    mask_to_pass_for_adaptive = cold_mask & (delta_sst_modified < threshold_min) & grad_mask
    adaptive_mask = adaptive_test(image=delta_sst_modified, window_size=window_size, scale=scale,
                                  mask=mask_to_pass_for_adaptive, threshold=threshold_min)
    # threshold_min = -1.5
    # threshold_max = -0.5
    return adaptive_mask & (delta_sst_modified < threshold_max)


def adaptive_boundaries_third_prototype(sst, cmc, threshold_min, threshold_max, threshold_gradient, radius, window_size,
                                        scale, cold_mask):
    delta_sst_original = sst - cmc
    grad_mask, cmc_min = high_gradient_sst(sst_reynolds=cmc, threshold=threshold_gradient, radius=radius)
    delta_sst_modified = sst - cmc_min
    mask_to_pass_for_adaptive = cold_mask & (delta_sst_modified < threshold_min)
    adaptive_mask = adaptive_test(image=delta_sst_original, window_size=window_size, scale=scale,
                                  mask=mask_to_pass_for_adaptive, threshold=threshold_min)
    # threshold_min = -1.5
    # threshold_max = -0.5
    return adaptive_mask & (delta_sst_original < threshold_max)


if __name__ == "__main__":
    print("File one executed when ran directly")
    path_tiles = r"C:\Users\Alexander Semenov\Desktop\ABI_ACSPO_CODE\cloudmasktest\tiles_pics\all_areas"
    path_abi = r"D:\Users\Alexander\ansel\TEMP\ABI\L2P"
    path_l1bs = r"D:\Users\Alexander\ansel\TEMP\ABI\Channel_10"
    files_l1bs = os.listdir(path_l1bs)
    path = path_abi
    files = os.listdir(path)
    files.sort()
    files_l1bs.sort()
    index = 5
    path_full = os.path.join(path, files[index])
    path_full_l1b = os.path.join(path_l1bs, files_l1bs[index])
    BT_7 = get_BT_from_L1b(path_full_l1b)
    print(files[index])
    # print(files_l1bs[0])
    sst, cmc = get_sst_cmc(path_full)
    mask = get_acspo_mask(path_full)
    BT_11 = get_Bt_from_L2P(path_full, "brightness_temp_ch14")
    BT_12 = get_Bt_from_L2P(path_full, "brightness_temp_ch15")
    BT_8 = get_Bt_from_L2P(path_full, "brightness_temp_ch11")
    BT_11_crtm = get_Bt_from_L2P(path_full, "brightness_temp_crtm_ch14")
    BT_12_crtm = get_Bt_from_L2P(path_full, "brightness_temp_crtm_ch15")
    BT_8_crtm = get_Bt_from_L2P(path_full, "brightness_temp_crtm_ch11")
    additional_mask = (BT_7 - BT_11 > -50)
    bt11_bt12_mask = np.abs((BT_11-BT_11_crtm) - (BT_12-BT_12_crtm))>  0.6
    bt12_bt8_mask = np.abs((BT_11-BT_11_crtm) - (BT_8-BT_8_crtm))>  1.2
    combined_mask = mask | (bt11_bt12_mask & ~np.isnan(sst))
    cmc_max, cmc_min = high_gradient_sst(sst_reynolds=cmc, threshold=2.0, radius=12)
    dt_sst = sst - cmc
    dt_sst_min = sst - cmc_min
    dt_sst_max = sst - cmc_max
    dt_sst[dt_sst > 2] = 2.0
    dt_sst[dt_sst < -2] = -2.0
    dt_sst_min[dt_sst_min > 2] = 2.0
    dt_sst_min[dt_sst_min < -2] = -2.0
    dt_sst_max[dt_sst_max > 2] = 2.0
    dt_sst_max[dt_sst_max < -2] = -2.0
    map_cloud = deepcopy(my_cmap)
    map_cloud.set_over((128 / 255.0, 128 / 255.0, 128 / 255.0))
    dt_sst[combined_mask] = 2.1
    dt_sst_min[mask] = 2.1
    dt_sst_max[mask] = 2.1
    sst[sst>298] = 298
    sst[combined_mask] = 300
    # slicing
    xmin = 600
    xmax = 920
    ymin = 2400
    ymax = 3200
    slice_1 = dt_sst[xmin:xmax, ymin:ymax]
    slice_2 = dt_sst_min[xmin:xmax, ymin:ymax]
    slice_3 = sst[xmin:xmax, ymin:ymax]
    slice_4 = cmc[xmin:xmax, ymin:ymax]
    slice_5 = dt_sst_max[xmin:xmax, ymin:ymax]
    # image range y - 700 - 1200, x - 2600 - 3300

    # draw_images((-2, 2, map_cloud), (-2, 2, my_cmap), (287, 302, my_cmap),(287, 302, my_cmap), dt_sst=slice_1, dt_min=slice_2, sst=slice_3,cmc=slice_4)
    draw_images((275, 298, map_cloud), sst=slice_3)
