import os
import h5py
import numpy
import pandas as pd
import tables
file_name  = "MYD03.A2021093.0210.061.NRT.hdf"

path = r"E:\input_vtune"

path_new = os.path.join(path,file_name)

data_set = pd.read_hdf(path_new,'r')

