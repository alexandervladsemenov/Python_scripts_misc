import netCDF4 as nc
import sys
import numpy as np
from MakeRGB import my_cmap
from matplotlib import pyplot as plt

from copy import deepcopy
map_cloud = deepcopy(my_cmap)
map_cloud.set_over((128 / 255.0, 128 / 255.0, 128 / 255.0))
path = r"D:\Users\Alexander\ansel\TEMP\2021-05-19\20210519120000-STAR-L3S_GHRSST-SSTsubskin-LEO_PM_D-ACSPO_V2.80-v02.0-fv01.0.nc"

data = nc.Dataset(path,"r")
print(data.variables.keys())
sst = np.array(data["sea_surface_temperature"][:]).reshape(9000,18000)
l2p_flags = np.array(data["l2p_flags"][:]).reshape(9000,18000)
sst[sst==-32768] = np.NaN
print(data.variables.keys())

cloud_mask = (l2p_flags==16896) |(l2p_flags==-7676)
sst[cloud_mask] = 600

plt.imshow(sst,interpolation="none",vmin=283,vmax=301,cmap=map_cloud)
# plt.imshow(l2p_flags, interpolation="none", cmap=my_cmap)
plt.colorbar()
plt.show()
while not plt.waitforbuttonpress(): pass
sys.exit()
