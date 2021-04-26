import cv2
import os
import numpy as np
import netCDF4 as nc
from matplotlib import pyplot as plt
import matplotlib
import sys


def print_ranks_false_alarms(differences_files: dict):
    for key in differences_files:
        print(differences_files[key][0], end='\t')
        for intems in differences_files[key][1]:
            print(intems, end='\t')
        print(key)


def validation(file_name: str):
    file_nc = nc.Dataset(file_name, "r")
    original_mask = np.array(file_nc.variables["original_mask"][:])
    validation_mask = np.array(file_nc.variables["validation_mask"][:])
    individual = np.array(file_nc.variables["individual_clear_sky_tests_results"][:])
    extra = np.array(file_nc.variables["extra_byte_clear_sky_tests_results"][:])
    land = np.isnan(np.array(file_nc.variables["sst_regression"][:]))
    ocean = ~ land
    static_mask = individual >> 2 & 1
    warm_static_mask = extra >> 0 & 1
    adaptive_mask = individual >> 3 & 1
    warm_adaptive_mask = extra >> 1 & 1
    pure_adaptive = adaptive_mask != static_mask
    pure_warm_adaptive = warm_adaptive_mask != warm_static_mask
    rgct_mask = individual >> 4 & 1
    rrct_mask = individual >> 5 & 1
    uniformity_mask = individual >> 6 & 1
    cross_corr_mask = individual >> 7 & 1
    pos_mask = extra >> 3 & 1

    total_mask = static_mask | pure_adaptive | rgct_mask | uniformity_mask | cross_corr_mask | pos_mask | pure_warm_adaptive | warm_static_mask
    diff = total_mask.astype(np.float32) - original_mask.astype(np.float32)
    sum = np.sum(diff)
    # if sum != 0:
    #     print(file_name,sum)
    diff_false = validation_mask.astype(np.float32) - total_mask.astype(np.float32) < 0
    diff_total = np.abs(validation_mask.astype(np.float32) - total_mask.astype(np.float32))
    diff_leakages = validation_mask.astype(np.float32) - total_mask.astype(np.float32) > 0

    diff_static = validation_mask.astype(np.float32) - static_mask.astype(np.float32) < 0
    diff_warm_static_mask = validation_mask.astype(np.float32) - warm_static_mask.astype(np.float32) < 0
    diff_adaptive_mask = validation_mask.astype(np.float32) - pure_adaptive.astype(np.float32) < 0
    diff_warm_adaptive_mask = validation_mask.astype(np.float32) - pure_warm_adaptive.astype(np.float32) < 0
    diff_rgct_mask = validation_mask.astype(np.float32) - rgct_mask.astype(np.float32) < 0
    diff_uniformity_mask = validation_mask.astype(np.float32) - uniformity_mask.astype(np.float32) < 0
    diff_cross_corr_mask = validation_mask.astype(np.float32) - cross_corr_mask.astype(np.float32) < 0
    diff_pos_mask = validation_mask.astype(np.float32) - pos_mask.astype(np.float32) < 0
    diff_rrct_mask = validation_mask.astype(np.float32) - rrct_mask.astype(np.float32) < 0

    static_correct = static_mask & (validation_mask == static_mask)
    warm_static_correct = warm_static_mask & (validation_mask == warm_static_mask)
    adaptive_correct = pure_adaptive & (pure_adaptive == validation_mask)
    warm_adaptive_correct = pure_warm_adaptive & (pure_warm_adaptive == validation_mask)
    uniformity_correct = uniformity_mask & (uniformity_mask == validation_mask)
    corr_corr_correct = cross_corr_mask & (cross_corr_mask == validation_mask)
    pos_correct = pos_mask & (pos_mask == validation_mask)
    rgct_correct = rgct_mask & (rgct_mask == validation_mask)
    rrct_correct = rrct_mask & (rrct_mask == validation_mask)
    total_correct = total_mask & (total_mask == validation_mask)

    inp_false = [np.sum(validation_mask), np.sum(total_mask), np.sum(diff_static), np.sum(diff_adaptive_mask),
                 np.sum(diff_warm_static_mask),
                 np.sum(diff_warm_adaptive_mask), np.sum(diff_rgct_mask), np.sum(diff_rrct_mask),
                 np.sum(diff_uniformity_mask),
                 np.sum(diff_cross_corr_mask), np.sum(diff_pos_mask), np.sum(ocean), np.sum(land)]

    inp_total = [np.sum(validation_mask), np.sum(total_mask), np.sum(static_mask), np.sum(pure_adaptive),
                 np.sum(warm_static_mask),
                 np.sum(pure_warm_adaptive), np.sum(rgct_mask), np.sum(rrct_mask),
                 np.sum(uniformity_mask),
                 np.sum(cross_corr_mask), np.sum(pos_mask), np.sum(ocean), np.sum(land)]

    inp_true = [ np.sum(validation_mask), np.sum(total_mask), np.sum(static_correct),
                np.sum(adaptive_correct), np.sum(warm_static_correct),
                np.sum(warm_adaptive_correct), np.sum(rgct_correct), np.sum(rrct_correct),
                np.sum(uniformity_correct),
                np.sum(corr_corr_correct), np.sum(pos_correct), np.sum(ocean), np.sum(land)]

    static_unique = static_mask & (static_mask != (
            pure_adaptive | rgct_mask | uniformity_mask | cross_corr_mask | pos_mask | pure_warm_adaptive | warm_static_mask))
    adaptive_unique = pure_adaptive & (pure_adaptive != (
            static_mask | rgct_mask | uniformity_mask | cross_corr_mask | pos_mask | pure_warm_adaptive | warm_static_mask))
    warm_static_unique = warm_static_mask & (warm_static_mask != (
            static_mask | pure_adaptive | rgct_mask | uniformity_mask | cross_corr_mask | pos_mask | pure_warm_adaptive))
    warm_adaptive_unique = pure_warm_adaptive & (pure_warm_adaptive != (
            static_mask | pure_adaptive | rgct_mask | uniformity_mask | cross_corr_mask | pos_mask | warm_static_mask))
    rgct_unique = rgct_mask & (rgct_mask != (
            static_mask | pure_adaptive | uniformity_mask | cross_corr_mask | pos_mask | pure_warm_adaptive | warm_static_mask))
    rrct_unique = rrct_mask
    uni_unqiue = uniformity_mask & (uniformity_mask != (
            static_mask | pure_adaptive | rgct_mask | cross_corr_mask | pos_mask | pure_warm_adaptive | warm_static_mask))
    cross_unique = cross_corr_mask & (cross_corr_mask != (
            static_mask | pure_adaptive | rgct_mask | uniformity_mask | pos_mask | pure_warm_adaptive | warm_static_mask))
    pos_unqiue =pos_mask & (pos_mask!=(static_mask | pure_adaptive | rgct_mask | uniformity_mask | cross_corr_mask | pure_warm_adaptive | warm_static_mask))

    inp_leakages = [np.sum(validation_mask), np.sum(total_mask), np.sum(ocean), np.sum(land)]

    inp_unique= [np.sum(validation_mask), np.sum(total_mask), np.sum(static_unique),
                np.sum(adaptive_unique), np.sum(warm_static_unique),
                np.sum(warm_adaptive_unique), np.sum(rgct_unique), np.sum(rrct_unique),
                np.sum(uni_unqiue),
                np.sum(cross_unique), np.sum(pos_unqiue), np.sum(ocean), np.sum(land)]

    inp_total_np = np.array(inp_total)
    inp_unique_np = np.array(inp_unique)
    inp_false_np = np.array(inp_false)
    inp_true_np = np.array(inp_true)
    unique_rat = inp_unique_np/ inp_total_np*100

    return np.sum(diff_total), unique_rat


path_main = r"C:\Users\Alexander Semenov\Desktop\ABI_ACSPO_CODE\cloudmasktest\tiles_pics"

region_names = ["abnormal_retrievals", "diurnal_warming", "dynamic_regions", "glint", "scattered_clouds"]
all_areas = "all_areas"
files_all_files = {}
differences_files = {}
counter = 0
static_arr: np.array = []
adaptive_arr: np.array = []
static_warm_arr: np.array = []
adaptive_warm_arr: np.array = []
rgct_arr: np.array = []
uni_arr: np.array = []
cross_arr: np.array = []
pos_arr: np.array = []
for name in region_names:

    folders = os.listdir(os.path.join(path_main, name))
    for folder in folders:
        if folder == all_areas:
            continue
        path = os.path.join(path_main, name, folder)
        files = os.listdir(path)
        files_all_files[name] = files
        for file in files:
            flag = True
            if ".nc" in file:
                flag = False
            if flag:
                continue
            path_file = os.path.join(path_main, name, folder, file)
            key = validation(path_file)[0]
            val = validation(path_file)[1]
            full_name = name + '/' + folder + "/" + file
            differences_files[full_name] = key, val
            static_arr.append(val[2])
            adaptive_arr.append(val[3])
            static_warm_arr.append(val[4])
            adaptive_warm_arr.append(val[5])
            rgct_arr.append(val[6])
            uni_arr.append(val[8])
            cross_arr.append(val[9])
            pos_arr.append(val[10])
            counter += 1

print_ranks_false_alarms(differences_files)

# sys.exit()

nb = 20
max_val = 100
array_bins  = np.arange(0.0, max_val + max_val/nb, max_val/nb)

plt.hist(static_arr, bins=array_bins, label='cold_static')
plt.hist(adaptive_arr, bins=array_bins, label='cold_adaptive')
# plt.hist(rgct_arr, bins=array_bins, label='rgct')
# plt.hist(static_warm_arr, bins=array_bins, label='warm_static')
# plt.hist(adaptive_warm_arr, bins=array_bins, label='warm_adaptive')
# plt.hist(cross_arr, bins=array_bins, label='cross_correlation')
# plt.hist(uni_arr, bins=array_bins, label='uniformity')
# plt.hist(pos_arr, bins=array_bins, label='positive_outlier')
plt.legend()
plt.show()


nb = 30
array_bins = np.arange(0.0, max_val + max_val/nb, max_val/nb)
plt.hist(rgct_arr, bins=array_bins, label='rgct')
plt.hist(adaptive_warm_arr, bins=array_bins, label='warm_adaptive')

plt.hist(static_warm_arr, bins=array_bins, label='warm_static')

# plt.hist(cross_arr, bins=array_bins, label='cross_correlation')
# plt.hist(uni_arr, bins=array_bins, label='uniformity')
# plt.hist(pos_arr, bins=array_bins, label='positive_outlier')
plt.legend()
plt.show()

nb = 40
array_bins = np.arange(0.0, max_val + max_val/nb, max_val/nb)


plt.hist(uni_arr, bins=array_bins, label='uniformity')

plt.hist(cross_arr, bins=array_bins, label='cross_correlation')
# plt.hist(pos_arr, bins=array_bins, label='positive_outlier')
plt.legend()
plt.show()




plt.hist(pos_arr, bins=array_bins, label='positive_outlier')
plt.legend()
plt.show()
sys.exit()
