import cv2
import os
import numpy as np
import netCDF4 as nc
from matplotlib import pyplot as plt
import sys
from numpy import linalg as la
from copy import deepcopy
from MakeRGB import my_cmap
from bt_tests import save_obj


def compute_covariance(input1, input2, rad):
    input1 = input1 - cv2.blur(input1, (rad, rad))
    input2 = input2 - cv2.blur(input2, (rad, rad))
    product = input1 * input2
    return cv2.blur(product, (rad, rad)) * (rad * rad) / (rad * rad - 1.0)


def compute_stats(mask_inp, rad):
    mask = (mask_inp).astype(np.float32)
    mask_to_do = cv2.blur(mask, (rad * 2, rad * 2)) == 1
    return mask_to_do & mask_inp


def compute_eigenvalues(data_vals: dict, rows, cols, eig_num=1):
    cov_size = len(data_vals)
    cov_matrix = np.empty((rows * cols, cov_size ** 2))
    i = -1

    for key_i in data_vals:
        i = i + 1
        j = -1
        var1 = data_vals[key_i]
        for key_j in data_vals:
            j = j + 1
            var2 = data_vals[key_j]
            index = i * cov_size + j
            cov_ij = compute_covariance(var1, var2, rad=5).reshape(rows * cols, )
            cov_matrix[:, index] = cov_ij

    cov_matrix = cov_matrix.reshape((rows * cols, cov_size, cov_size))
    w, v = la.eig(cov_matrix)

    ww = -np.sort(-w)

    ww = ww.reshape((rows, cols, cov_size))

    return ww


path = r"C:\Users\Alexander Semenov\Desktop\ABI_ACSPO_CODE\cloudmasktest\tiles_pics\all_areas"

files = os.listdir(path)
files.sort()
key_names = ["brightness_temp_ch7", "brightness_temp_ch11", "brightness_temp_ch14", "brightness_temp_ch15","brightness_temp_ch13",
             "sst_regression"]

data_save_to_file = {}
data_save_to_file["only_bts"] = np.empty((68, 256, 256, len(key_names)-1))
data_save_to_file["sst_bts"] = np.empty((68, 256, 256, len(key_names)))
data_save_to_file["validation_mask"] = np.zeros((68, 256, 256)) > 1.0
data_save_to_file["no_mask"] = np.zeros((68, 256, 256)) > 1.0

map_cloud = deepcopy(my_cmap)
map_cloud.set_over((128 / 255.0, 128 / 255.0, 128 / 255.0))

for i in range(68):
    file = files[i]

    path_file = os.path.join(path, file)
    data = nc.Dataset(path_file, mode="r")
    data_vals = {}
    validation_mask = np.array(data.variables["validation_mask"][:]).astype(np.bool)
    rows, cols = validation_mask.shape
    data_save_to_file["validation_mask"][i, :, :] = validation_mask


    for key in key_names:
        data_vals[key] = np.array(data[key][:])  # 4nm
        mask = np.isnan(data_vals[key])
        val = np.nanmedian(data_vals[key])
        data_vals[key][mask] = val
    mask_nan = ~np.isnan(np.array(data["sst_regression"][:]))
    data.close()

    image = compute_eigenvalues(data_vals=data_vals, rows=rows, cols=cols)

    mask_to_do = compute_stats(mask_inp=mask_nan, rad=5)
    data_vals.popitem()
    image2 = compute_eigenvalues(data_vals=data_vals, rows=rows, cols=cols)
    data_save_to_file["sst_bts"][i, :, :, :] = image
    data_save_to_file["only_bts"][i, :, :, :] = image2
    data_save_to_file["no_mask"][i, :, :] = mask_to_do
save_obj(obj=data_save_to_file, name="covariance")
sys.exit()

# [0.00070906 0.00128964 0.00323826 0.02029807]
# [0.00026574 0.00071616 0.00293377 0.00896228 0.02609597]
