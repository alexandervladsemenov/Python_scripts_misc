import cv2
import os
import numpy as np
import netCDF4 as nc
from matplotlib import pyplot as plt
import sys
from MakeRGB import my_cmap

path = r"C:\Users\Alexander Semenov\Desktop\ABI_ACSPO_CODE\cloudmasktest\tiles_pics\all_areas"

files = os.listdir(path)
num_files = len(files)

bt11 = np.empty((num_files, 256, 256))
bt11_crtm = np.empty((num_files, 256, 256))
bt12 = np.empty((num_files, 256, 256))
bt12_crtm = np.empty((num_files, 256, 256))
delta = np.empty((num_files, 256, 256))
i = 0
for file in files:
    data = nc.Dataset(os.path.join(path, file), "r")
    solar_zenith_angle = np.array(data["solar_zenith_angle"][:])
    brightness_temp_ch7 = np.array(data["brightness_temp_ch7"][:])  # 4nm
    brightness_temp_ch11 = np.array(data["brightness_temp_ch11"][:])  # 8nm
    brightness_temp_ch14 = np.array(data["brightness_temp_ch14"][:])  # 11nm
    brightness_temp_ch15 = np.array(data["brightness_temp_ch15"][:])  # 12nm
    individual = np.array(data["individual_clear_sky_tests_results"][:])
    atm_diff = np.array(data.variables["atm_diff"][:])
    sst_regression = np.array(data["sst_regression"][:])
    sst_reynolds = np.array(data["sst_reynolds"][:])
    brightness_temp_crtm_ch7 = np.array(data["brightness_temp_crtm_ch7"][:])
    brightness_temp_crtm_ch11 = np.array(data["brightness_temp_crtm_ch11"][:])
    brightness_temp_crtm_ch14 = np.array(data["brightness_temp_crtm_ch14"][:])
    brightness_temp_crtm_ch15 = np.array(data["brightness_temp_crtm_ch15"][:])
    validation_mask = np.array(data["validation_mask"][:]).astype(bool)
    original_mask = np.array(data["original_mask"][:]).astype(bool)
    glint = np.array(data["glint"][:]).astype(bool)
    data.close()
    bt11[i, :, :] = brightness_temp_ch14
    bt11_crtm[i, :, :] = brightness_temp_crtm_ch14
    bt12[i, :, :] = brightness_temp_ch15
    bt12_crtm[i, :, :] = brightness_temp_crtm_ch15
    i += 1

rad = 3

for i in range(num_files):
    delta[i, :, :] = cv2.GaussianBlur(bt11[i, :, :] - bt12[i, :, :] - bt11_crtm[i, :, :] + bt12_crtm[i, :, :],
                                      (rad, rad), 0)
    delta[i, :, :] = bt11[i, :, :] - bt12[i, :, :] - bt11_crtm[i, :, :] + bt12_crtm[i, :, :]

data_non_nan = delta[~np.isnan(delta)]

plt.hist(data_non_nan,bins=256)

plt.show()

plt.waitforbuttonpress()
sys.exit()
