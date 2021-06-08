import cv2
import pickle
import numpy as np
import os
import netCDF4 as nc
import sys
from scipy import stats
from MakeRGB import my_cmap
from matplotlib import pyplot as plt
from copy import deepcopy
from algorithm import get_BTS, dict_BTS_9, dict_BTS_10
from scipy.stats import pearsonr
from scipy import signal
from adaptive_test import adaptive_test
from bt_tests import get_high_gradient,high_gradient_sst

path_to_granules = r"D:\Users\Alexander\Downloads\granules_L2P_test"



files = os.listdir(path_to_granules)
print(files)
threshold = 2
radius = 15

file = files[1]

path_to_file = os.path.join(path_to_granules, file)
nc_data = nc.Dataset(path_to_file, mode="r")
print(nc_data.variables)

min_sst = -32768
sst = np.array(nc_data["sea_surface_temperature"][:, :])
sst_ref = sst.reshape(5424, 5424)
sst_ref[sst_ref == min_sst] = np.NaN
nan_mask = np.isnan(sst_ref)
d_sst = np.array(nc_data["dt_analysis"][:, :])

delta_sst = d_sst.reshape(5424, 5424)
delta_sst[nan_mask] = np.NaN






plt.imshow(delta_sst,interpolation="none",vmin=-2,vmax=2,cmap=my_cmap)






plt.colorbar()

plt.show()
plt.waitforbuttonpress()
# 1.1218497743522697 for 2
# 1.1144564690991983 for 1
# tiles 6.106913272595465
# 6.106913272595465
sys.exit()
