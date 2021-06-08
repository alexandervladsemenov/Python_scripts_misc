import sys

import cv2
import os
import numpy as np
import netCDF4 as nc
from MakeRGB import my_cmap
from matplotlib import pyplot as plt

path = r"C:\Users\Alexander Semenov\Desktop\ABI_ACSPO_CODE\cloudmasktest\tiles_pics\all_areas"

files = os.listdir(path)

files.sort()
i = 4
for file in files[i:i+1]:
    path_full = os.path.join(path, file)
    data = nc.Dataset(path_full, mode="r")
    sst_regression = np.array(data["sst_regression"][:])
    sst_reynolds = np.array(data["sst_reynolds"][:])
    vmin = np.nanmin(sst_reynolds)
    vmax = np.nanmax(sst_reynolds)
    delta_sst = sst_regression - sst_reynolds
    sst =  cv2.medianBlur(sst_regression.astype(np.float32), 5, 0)
    plt.imshow(sst,interpolation="none",vmin=vmin-2,vmax=vmax,cmap=my_cmap)
    plt.colorbar()
    plt.show()
    plt.waitforbuttonpress()
    plt.close()
    plt.imshow(sst_reynolds,interpolation="none",vmin=vmin,vmax=vmax,cmap=my_cmap)
    plt.colorbar()
    plt.show()
    plt.waitforbuttonpress()


sys.exit()

