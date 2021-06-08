import os

import netCDF4 as nc
from plot_covariance import plot_hist
import numpy as np
from MakeRGB import my_cmap
from copy import deepcopy
from adaptive_modified import draw_images, get_BT_from_L1b
from matplotlib import pyplot as plt
from bt_tests import Plank_function

key_words = ["solar_zenith_angle", "satellite_zenith_angle", "brightness_temp_ch7", "sst_regression", "sst_reynolds",
             "brightness_temp_ch11", "brightness_temp_ch14", "brightness_temp_ch15", "brightness_temp_crtm_ch7",
             "brightness_temp_crtm_ch11", "brightness_temp_crtm_ch14", "brightness_temp_crtm_ch15", "acspo_mask"
             ]
large_flag = True


def add_bt_10(paths_l1b, training_data, size=5424):
    training_data["brightness_temp_ch10"] = np.zeros((size, size, len(paths))).astype(np.float32)
    index = 0
    for path in paths_l1b:
        training_data["brightness_temp_ch10"][:, :, index] = get_BT_from_L1b(path)
        index += 1


def create_validation_base(paths, size=256):
    training_data = {}
    for word in key_words:
        if word == "acspo_mask":
            training_data[word] = np.zeros((size, size, len(paths))).astype(np.uint8)
        else:
            training_data[word] = np.zeros((size, size, len(paths))).astype(np.float32)
    index = 0
    training_data["valid"] = np.zeros((size, size, len(paths))) > 1.0
    for path in paths:
        data = nc.Dataset(path, mode="r")
        for word in key_words:
            if word == "acspo_mask":
                training_data[word][:, :, index] = np.array(data.variables[word][:]).astype(np.uint8)
            else:
                training_data[word][:, :, index] = np.array(data.variables[word][:]).astype(np.float32)
        try:
            training_data["valid"][:, :, index] = np.array(data.variables["validation_mask"][:]).astype(np.bool)
        except:
            print("not found")
        index = index + 1
    return training_data


if __name__ == "__main__":
    map_cloud = deepcopy(my_cmap)
    map_cloud.set_over((128 / 255.0, 128 / 255.0, 128 / 255.0))
    map_cloud.set_under((255 / 255.0, 105 / 255.0, 180 / 255.0))  # hot pink

    if large_flag:
        path_abi = r"D:\Users\Alexander\ansel\TEMP\ABI\L2P"
    else:
        path_abi = r"C:\Users\Alexander Semenov\Desktop\ABI_ACSPO_CODE\cloudmasktest\tiles_pics\all_areas"

    path_l1bs = r"D:\Users\Alexander\ansel\TEMP\ABI\Channel_10"
    file_l1bs = os.listdir(path_l1bs)
    files = os.listdir(path_abi)
    files.sort()
    file_l1bs.sort()
    num = len(files)
    paths = []
    paths_l1bs = []
    for file in files:
        paths.append(os.path.join(path_abi, file))
    for file in file_l1bs:
        paths_l1bs.append(os.path.join(path_l1bs, file))
    print(paths_l1bs)
    size = 256
    if large_flag:
        size = 5424

    training_data = create_validation_base(paths, size=size)
    add_bt_10(paths_l1bs,training_data,size=size)

    # create a validation mask

    training_data["validation_mask"] = np.zeros((size, size, len(paths))) > 1.0
    training_data["BT11_BT12"] = np.zeros((size, size, len(paths))).astype(np.float32)
    print(num, len(training_data["acspo_mask"]))
    if not large_flag:
        training_data["validation_mask"] = training_data["valid"]
    else:
        training_data["validation_mask"] = ((training_data["acspo_mask"] >> 7 & 1) > 0) & ~(
                (training_data["acspo_mask"] >> 6 & 1) > 0)

    training_data["BT11_BT12"] = training_data["brightness_temp_ch14"] - training_data["brightness_temp_ch15"] + \
                                 training_data["brightness_temp_crtm_ch15"] - training_data["brightness_temp_crtm_ch14"]
    training_data["BT12_BT8"] = training_data["brightness_temp_ch11"] - training_data["brightness_temp_ch15"] + \
                                training_data["brightness_temp_crtm_ch15"] - training_data["brightness_temp_crtm_ch11"]

    rad_rat_observed = Plank_function(training_data["brightness_temp_ch7"]) / Plank_function(
        training_data["brightness_temp_ch14"])
    rad_rat_clear_Sky = Plank_function(training_data["brightness_temp_crtm_ch7"]) / Plank_function(
        training_data["brightness_temp_crtm_ch14"])

    solzen_mask_night = training_data["solar_zenith_angle"] < 90.0  # day

    ulst = rad_rat_clear_Sky - rad_rat_observed
    ulst[solzen_mask_night] = np.NaN
    training_data["ULST"] = ulst
    training_data["BT11_BT7"] = training_data["brightness_temp_ch10"] -  training_data["brightness_temp_ch14"]
    data_clear = {}
    data_cloudy = {}

    satzen = np.abs(training_data["satellite_zenith_angle"])

    data_clear["var"] = training_data["BT11_BT12"]
    data_clear["mask"] = (training_data["validation_mask"] == 0) & ~np.isnan(training_data["sst_regression"]) & (
            (training_data["sst_regression"] -
             training_data["sst_reynolds"]) < 0.0) & (satzen < 57)
    data_clear["title"] = "Clear"

    data_cloudy["var"] = training_data["BT11_BT12"]
    data_cloudy["mask"] = (training_data["validation_mask"] > 0) & (
            (training_data["sst_regression"] -
             training_data["sst_reynolds"]) < 0.0) & (satzen < 57)
    data_cloudy["title"] = "Cloudy"

    plot_hist(xmin=-55, xmax=-45, nbins=200, data_clear=data_clear, data_cloudy=data_cloudy)

    plot_figures = False

    if plot_figures:
        new_mask = (training_data["BT11_BT12"] < -1.2) & (training_data["validation_mask"] == 0) & ~np.isnan(
            training_data["sst_regression"])

        fig_index = 1

        delta_sst = training_data["sst_regression"][:, :, fig_index] - training_data["sst_reynolds"][:, :, fig_index]

        sat_zen = np.abs(training_data["satellite_zenith_angle"][:, :, fig_index])

        delta_sst[delta_sst > 2] = 1.9995
        delta_sst[delta_sst < -2] = -1.9995
        pure_dt = deepcopy(delta_sst)
        delta_sst[training_data["validation_mask"][:, :, fig_index] > 0] = 2.1
        delta_sst[new_mask[:, :, fig_index]] = - 2.1
        draw_images((-2, 2, map_cloud), (-2, 2, my_cmap), (0, 58, my_cmap), delta_sst=delta_sst, sst=pure_dt,
                    sat_zen=sat_zen)
