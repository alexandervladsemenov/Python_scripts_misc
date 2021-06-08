import numpy as np
import netCDF4 as nc
from matplotlib import pyplot as plt
import os
import sys
import copy


path = r"C:\Users\Alexander Semenov\Desktop\ABI_ACSPO_CODE\cloudmasktest\tiles_pics\all_areas"

files = os.listdir(path)

validation = np.zeros((68, 256, 256)) > 1
sat_zen = np.empty((68, 256, 256))
sst_bt11 = np.empty((68, 256, 256))
bt11_bt12 = np.empty((68, 256, 256))
bt11_bt8 = np.empty((68, 256, 256))
delta_sst = np.empty((68, 256, 256))
files.sort()

i = 0
for file in files:
    data = nc.Dataset(os.path.join(path, file), "r")
    satzen = np.array(data["satellite_zenith_angle"][:])
    day_mask = np.array(data["solar_zenith_angle"][:]) < 90.0
    sst_regression = np.array(data["sst_regression"][:])
    sst_reynolds = np.array(data["sst_reynolds"][:])
    validation_mask = np.array(data["validation_mask"][:]).astype(np.bool)
    brightness_temp_ch14 = np.array(data["brightness_temp_ch14"][:])
    brightness_temp_crtm_ch14 = np.array(data["brightness_temp_crtm_ch14"][:])
    brightness_temp_ch15 = np.array(data["brightness_temp_ch15"][:])
    brightness_temp_crtm_ch15 = np.array(data["brightness_temp_crtm_ch15"][:])
    brightness_temp_crtm_ch11 = np.array(data["brightness_temp_crtm_ch11"][:])
    brightness_temp_ch11 = np.array(data["brightness_temp_ch11"][:])  # 8nm

    data.close()
    dt_sst_bt11 = sst_regression - sst_reynolds - (brightness_temp_ch14 - brightness_temp_crtm_ch14)
    dt_bt11_bt12 = (brightness_temp_ch14 - brightness_temp_crtm_ch14) - (
            brightness_temp_ch15 - brightness_temp_crtm_ch15)
    dt_bt11_b8 = (brightness_temp_ch14 - brightness_temp_crtm_ch14) - (
            brightness_temp_ch11 - brightness_temp_crtm_ch11)
    delta_sst_local = sst_regression - sst_reynolds
    dt_bt11_b8 = brightness_temp_ch11 - brightness_temp_ch14
    dt_bt11_bt12 = brightness_temp_ch14 - brightness_temp_ch15
    validation[i, :, ::] = validation_mask
    sat_zen[i, :, :] = satzen
    dt_sst_bt11[day_mask] = np.NaN
    sst_bt11[i, :, :] = dt_sst_bt11
    delta_sst[i, :, :] = delta_sst_local
    bt11_bt12[i, :, :] = dt_bt11_bt12
    bt11_bt8[i, :, :] = dt_bt11_b8

    i = i + 1
quantity = bt11_bt8
parameter = bt11_bt12
mask = (~np.isnan(quantity)) & (~validation) & (~np.isnan(parameter))  # & (delta_sst > 0)

x_data = parameter[mask]
y_data = quantity[mask]
clear = copy.deepcopy(y_data)

plt.hist2d(x_data, y_data, bins=(40, 40), vmax=200, cmap="jet")
plt.xlabel("parameter")
plt.ylabel("quantity")
plt.title("quantity")
plt.colorbar()
plt.show()

mask = (~np.isnan(quantity)) & (validation) & (~np.isnan(parameter))  # & (delta_sst > 0)

x_data = parameter[mask]
y_data = quantity[mask]
clouds = copy.deepcopy(y_data)

plt.hist2d(x_data, y_data, bins=(40, 40), vmax=200, cmap="jet")
plt.xlabel("parameter")
plt.ylabel("quantity")
plt.title("Clouds")
plt.colorbar()
plt.show()

# nb = 200
# max_val = 2.0
# min_val = -3
# step = (max_val - min_val) / nb
# array_bins = np.arange(min_val, max_val + max_val / nb, step)
#
# plt.hist(clouds, color="blue", bins=array_bins, ls='dashed', lw=3, fc=(0, 0, 1, 0.5), label='Cloudy')
# plt.hist(clear, color="red", bins=array_bins, ls='dotted', lw=3, fc=(1, 0, 0, 0.5), label='Clear')
# plt.legend()
# plt.title("Clear/Cloudy")
# plt.xlabel("quantity")
# plt.ylabel("Number of pixels")
# plt.show()

sys.exit()
