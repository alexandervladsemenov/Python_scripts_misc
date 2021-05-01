import numpy as np
import netCDF4 as nc
from matplotlib import pyplot as plt
from matplotlib import cm
import os
import sys
import copy
from my_map import my_cmap
from matplotlib import patches

path = r"C:\Users\Alexander Semenov\Desktop\ABI_ACSPO_CODE\cloudmasktest\tiles_pics\all_areas"

files = os.listdir(path)

validation = np.zeros((68, 256, 256)) > 1
sat_zen = np.empty((68, 256, 256))
sst_bt11 = np.empty((68, 256, 256))
delta_sst = np.empty((68, 256, 256))
files.sort()

i = 0
for file in files:
    data = nc.Dataset(os.path.join(path, file), "r")
    satzen = np.array(data["satellite_zenith_angle"][:])
    sst_regression = np.array(data["sst_regression"][:])
    sst_reynolds = np.array(data["sst_reynolds"][:])
    validation_mask = np.array(data["validation_mask"][:]).astype(np.bool)
    brightness_temp_ch14 = np.array(data["brightness_temp_ch14"][:])
    brightness_temp_crtm_ch14 = np.array(data["brightness_temp_crtm_ch14"][:])
    data.close()
    dt_sst_bt11 = sst_regression - sst_reynolds - (brightness_temp_ch14 - brightness_temp_crtm_ch14)
    delta_sst_local = sst_regression - sst_reynolds
    validation[i, :, ::] = validation_mask
    sat_zen[i, :, :] = satzen
    sst_bt11[i, :, :] = dt_sst_bt11
    delta_sst[i, :, :] = delta_sst_local

    i = i + 1

mask = (~np.isnan(sst_bt11)) & (~validation) # & (delta_sst > 0)

x_data = sat_zen[mask]
y_data = sst_bt11[mask]
clear = copy.deepcopy(y_data)

plt.hist2d(x_data, y_data, bins=(40, 40), vmax=1000, cmap="jet")
plt.xlabel("VZA")
plt.ylabel("d_sst_bt11,K")
plt.title("Clear Sky")
plt.colorbar()
plt.show()

mask = (~np.isnan(sst_bt11)) & (validation)# & (delta_sst > 0)

x_data = sat_zen[mask]
y_data = sst_bt11[mask]
clouds = copy.deepcopy(y_data)

plt.hist2d(x_data, y_data, bins=(40, 40), vmax=1000, cmap="jet")
plt.xlabel("VZA")
plt.ylabel("d_sst_bt11,K")
plt.title("Clouds")
plt.colorbar()
plt.show()

nb = 200
max_val = 12
min_val = -5
step = (max_val-min_val)/nb
array_bins = np.arange(min_val, max_val + max_val / nb, step)

plt.hist(clouds, color="blue", bins=array_bins, ls='dashed', lw=3, fc=(0, 0, 1, 0.5), label='Cloudy')
plt.hist(clear, color="red", bins=array_bins, ls='dotted', lw=3, fc=(1, 0, 0, 0.5), label='Clear')
plt.legend()
plt.title("Clear/Cloudy")
plt.xlabel("d_sst_bt11,K")
plt.ylabel("Number of pixels")
plt.show()
sys.exit()
