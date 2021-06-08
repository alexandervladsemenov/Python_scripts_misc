import cv2
import os
import numpy as np
import netCDF4 as nc
from matplotlib import pyplot as plt
import sys
import matplotlib
import seaborn as sns
from scipy.stats import gaussian_kde
from MakeRGB import my_cmap
from scipy import stats
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
from adaptive_test import adaptive_test
from adaptive_modified import adaptive_boundaries_first_prototype, adaptive_boundaries_second_prototype, \
    adaptive_boundaries_third_prototype
from bt_tests import harmonics_cuts, uniformity_test, save_obj, load_obj, compute_BTDs_threholds, output_tiles_adaptive, \
    output_tiles_BTDS, rankings_of_btd, remove_files, get_high_gradient, temperature_flags_list, ranking_individual_btd, \
    draw_a_mask_upon, get_cold_sst_matrix, high_gradient_sst, output_files_sst_BTD, print_locations, mask_around

from plot_covariance import plot_hist

switch_full_btds = False
switch_adaptive = True


# def Plank_function(T, wave=3.9):
#     c2 = 1.4387752 * 1e4
#     return 1.0 / (np.exp(c2 / T / wave) - 1.0)


def get_threshold_hist(btd, validation, xmax, title, additional_mask, xmin=0,
                       path=r"D:\Users\Alexander\ACSPO\Opencv\Thresholds"):
    path_to_save = path
    validation_mask = validation > 0
    nb = 200
    plt_label_x = "BTD,K"
    plt_label_y = "Probability Density per K"
    list_of_radiance_tests = ["EMISS4_GLINT", "EMISS4", "ULST"]
    if title in list_of_radiance_tests:
        plt_label_x = "Relative_Radiance"
        plt_label_y = "Probability Density per Relative_Radiance"
    if title == "RGCT":
        return
    if (title in ["BT11_BT4", "BT12_BT4"]):
        xmin = -4
    if (title in ["BT11_BT8", "BT12_BT8"]):
        xmin = -2

    if title == "BT11_BT7":
        xmin = -60
        xmax = -50
    if title == "delta_sst":
        xmin = -10
        xmax = -.025

    max_val = xmax - xmin
    array_bins = np.arange(xmin, xmax + max_val / nb, max_val / nb)
    plt.hist(btd[validation_mask & ~np.isnan(btd) & additional_mask], color="blue", bins=array_bins, label='Cloudy')
    plt.hist(btd[~validation_mask & ~np.isnan(btd) & additional_mask], color="red", bins=array_bins, label='Clear')
    plt.legend()
    plt.title("Test = {} clear/cloudy".format(title))
    plt.ylabel(plt_label_y)
    plt.xlabel(plt_label_x)
    plt.show()
    plt.waitforbuttonpress()
    plt.savefig(os.path.join(path_to_save, title))
    plt.close()


path = r"C:\Users\Alexander Semenov\Desktop\ABI_ACSPO_CODE\cloudmasktest\tiles_pics\all_areas"

files = os.listdir(path)

test_names_original = ["PFMFT_BT11_BT12", "NFMFT_BT11_BT12", "BT11_BT8", "BT12_BT8", "BT11_BT4", "BT12_BT4",
                       "EMISS4", "SST_DCT", "UNI_SST", "EMISS4_GLINT", "RGCT",
                       "RFMFT_BT11_BT12",
                       "ULST", "ULST_COLD"]
test_names = ["UNI_SST", "RGCT", "ULST", "SST_BT11", "SST_BT4"]  # "BT11_BT8", "PFMFT_BT11_BT12"

print("number of tests", len(test_names))
try:
    tests_data = load_obj("data")
except:
    # validation mask
    tests_data: dict = {}
    tests_data["Validation"] = np.empty((len(files), 256, 256))

    # static mask mask

    tests_data["Static"] = np.empty((len(files), 256, 256))

    # adaptive mask

    tests_data["Adaptive"] = np.empty((len(files), 256, 256))

    # original mask

    tests_data["Original"] = np.empty((len(files), 256, 256))

    # delta sst
    tests_data["delta_sst"] = np.empty((len(files), 256, 256))

    # Individual
    tests_data["Individual"] = np.empty((len(files), 256, 256))

    # solzen
    tests_data["solzen"] = np.empty((len(files), 256, 256))

    # sst_reynolds
    tests_data["sst_reynolds"] = np.empty((len(files), 256, 256))

    # sst_regression
    tests_data["sst_regression"] = np.empty((len(files), 256, 256))

    # non_sst_mask
    tests_data["non_sst_mask"] = np.empty((len(files), 256, 256))
    # UNIFORM
    tests_data["UNIFORM"] = np.empty((len(files), 256, 256))
# add_names = ["UNI_SST", "RGCT", "PFMFT_BT11_BT12", "NFMFT_BT11_BT12", "BT11_BT8", "BT12_BT8", "BT11_BT4", "BT12_BT4",
#              "EMISS4", "EMISS4_GLINT",
#              "RFMFT_BT11_BT12", "SST_BT12", "SST_BT11",
#              "ULST"]

files.sort()
# output_file_name = "./composite_btds/locations.dat"
# print_locations(files, path, output_file_name)
add_names = ["BT12_BT8"]
# for test_name in add_names:
#     tests_data[test_name] = np.empty((len(files), 256, 256))
# tests_data = compute_BTDs_threholds(files=files, path=path, tests_data=tests_data, test_names=add_names)
# save_obj(tests_data, "data")

# sys.exit()

x_range = {}

x_range["PFMFT_BT11_BT12"] = 2.0
x_range["NFMFT_BT11_BT12"] = 1.0
x_range["RFMFT_BT11_BT12"] = 1.5
x_range["BT11_BT8"] = 2
x_range["BT12_BT8"] = 2
x_range["BT11_BT4"] = 4
x_range["BT12_BT4"] = 4
x_range["BT11STD"] = 2.0
x_range["BT12_BT4_NIGHT"] = 13
x_range["BT12_BT4_DAY"] = 18
x_range["SST_BT11"] = 7.
x_range["SST_BT12"] = 8.
x_range["SST_DCT"] = 2
x_range["SST_DCT_MEDIAN"] = 2
x_range["ULST"] = 0.25
x_range["UNI_SST"] = 1
x_range["UNI_SST2"] = 1
x_range["UNI_ULST"] = 1
x_range["UNI_EMISS4"] = 0.5
x_range["EMISS4_GLINT"] = 5
x_range["RGCT"] = 5
x_range["EMISS4"] = 0.5
x_range["WATER_VAPOR_TEST"] = 0.5
x_range["WATER_VAPOR_TEST_10"] = 0.4
x_range["LAPLACE"] = 30
x_range["PEARSON"] = 1.0
x_range["RTCT"] = 2
x_range["N_OTC"] = 20
x_range["BT11_BT7"] = 60
x_range["delta_sst"] = -6
# for test_name in add_names:
#     print(test_name)
#     additional_mask = tests_data["delta_sst"] > 0
#     get_thresold_hist(tests_data[test_name], tests_data["Validation"], xmax=x_range[test_name], title=test_name,
#                       additional_mask=additional_mask, path="./Thresh_warm/")  #
# for test_name in add_names:
#     print(test_name)
#     additional_mask = tests_data["delta_sst"] <= 0
#     get_threshold_hist(tests_data[test_name], tests_data["Validation"], xmax=x_range[test_name], title=test_name,
#                        additional_mask=additional_mask, path="./Thresh_cold")  #

# sys.exit()


thresholds = {}
thresholds2 = {}
test_for_second_threshold = ["BT11_BT8", "BT12_BT8", "BT11_BT4", "BT12_BT4"]
thresholds["PFMFT_BT11_BT12"] = 0.8  # 0.8
thresholds["NFMFT_BT11_BT12"] = 0.5
thresholds["RFMFT_BT11_BT12"] = 0.7
thresholds["RTCT"] = 0.5

thresholds["BT11_BT8"] = 1.6
thresholds["BT12_BT8"] = 1.3
thresholds["BT11_BT4"] = 3.0
thresholds["BT12_BT4"] = 4.0

thresholds2["BT11_BT8"] = -0.9
thresholds2["BT12_BT8"] = -1.3
thresholds2["BT11_BT4"] = -2.5
thresholds2["BT12_BT4"] = -4.0

thresholds["SST_BT11"] = 4.6  # 4.5
thresholds["SST_BT12"] = 4.5  #
thresholds["SST_BT4"] = 3.1  # 3.2

thresholds["ULST"] = 0.08  # 0.06 # 0.08
thresholds["UNI_SST"] = 0.05
thresholds["EMISS4"] = 0.15
thresholds["RGCT"] = 0.0
thresholds["BT11STD"] = 1.1

adaptive_radius = {}
adaptive_scale = {}

for key in thresholds:
    adaptive_radius[key] = -5
    adaptive_scale[key] = 6

tests_data["FULL_BTD_MASK"] = np.zeros((68, 256, 256)) > 1.0

adaptive_radius["SST_BT11"] = 11
adaptive_radius["SST_BT4"] = 5
# adaptive_radius["ULST"] = 10
# adaptive_scale["ULST"] = 3
# adaptive_radius["PFMFT_BT11_BT12"] = 5
# adaptive_scale["PFMFT_BT11_BT12"] = 3
warm_mask_threshold = 0.5
for test_name in test_names:
    radius = adaptive_radius[test_name]
    scale = adaptive_scale[test_name]
    threshold = thresholds[test_name]
    tests_data[test_name + "_adaptive"] = np.empty((68, 256, 256))
    tests_data[test_name + "_static"] = np.empty((68, 256, 256))

    for i in range(len(files)):
        image = tests_data[test_name][i, :, :]
        mask = (image > threshold) & (~np.isnan(image))
        if test_name in temperature_flags_list:
            mask = (mask & (tests_data["delta_sst"][i, :, :] > warm_mask_threshold))
            adapt_mask = adaptive_test(image=image, window_size=radius, scale=scale,
                                       threshold=threshold, mask=mask) & (~np.isnan(image))
            mask = mask | adapt_mask
            if test_name in test_for_second_threshold:
                threshold2 = thresholds2[test_name]
                mask2 = (image < threshold2) & (~np.isnan(image)) & (
                        tests_data["delta_sst"][i, :, :] > warm_mask_threshold)
                mask = mask2 | mask  # need to separate later
        tests_data[test_name + "_static"][i, :, :] = mask

        # disable adaptive test
        # radius = -5
        # tests_data[test_name + "_adaptive"][i, :, :] = adaptive_test(image=image, window_size=radius, scale=scale,
        #                                                              threshold=threshold, mask=mask) & (
        #                                                    ~np.isnan(image))
        tests_data["FULL_BTD_MASK"][i, :, :] = tests_data["FULL_BTD_MASK"][i, :, :] | (
                tests_data[test_name + "_static"][i, :, :] > 0)  # (
        # tests_data[test_name + "_adaptive"][i, :, :] > 0) |

thresh = 2.5  # 2.5 for min_max
dilatation_size = 11  # 9 for min_max square window, 11 for circle

high_grad_masks = {}

test_names_cold = ["NFMFT_BT11_BT12", "BT12_BT8",
                   "ULST_COLD", "RTCT", "PFMFT_BT11_BT12", "BT11_BT7"
                   ]  # "RTCT","BT11STD" "BT11_BT8", "BT11_BT4", ,"ULST_COLD", "RFMFT_BT11_BT12", "covariance"

test_dt_threshold = []

cold_masks, cold_mask_by_test = get_cold_sst_matrix(tests_data, files, test_names_cold, test_dt_threshold)

for test_cold_name in test_names_cold:
    tests_data[test_cold_name + "_static"] = np.zeros((len(files), 256, 256)) > 1.0

land_rad = 10

tests_data["ACSPO_mod" + "_static"] = np.zeros((len(files), 256, 256)) > 1.0
tests_data["GRAD_MASK"] = np.zeros((len(files), 256, 256)) > 1.0

for i in range(len(files)):
    individual_original = tests_data["Individual"][i, :, :]
    sst_reynolds = tests_data["sst_reynolds"][i, :, :]
    sst_regression = tests_data["sst_regression"][i, :, :]
    nan_mask = np.isnan(sst_regression)
    mask_no_land = (mask_around(nan_mask, land_rad)) & (~nan_mask)
    hmask = get_high_gradient(sst_reynolds, threshold=thresh, radius=dilatation_size)
    # other mask
    sst_max, sst_min = high_gradient_sst(sst_reynolds, threshold=thresh, radius=dilatation_size+3)

    cold_mask = cold_masks[files[i]] > 0  #
    mask_to_protect = (tests_data["FULL_BTD_MASK"][i, :, :] > 0) | cold_mask

    non_sst_mask = tests_data["non_sst_mask"][i, :, :]
    tests_data["GRAD_MASK"][i, :, :] = hmask
    if switch_full_btds:
        hmask[:] = True
    Adaptive = ((tests_data["Adaptive"][i, :, :] > 0) & (~hmask))  # it could be that we would need to return hmask

    for test_cold_name in test_names_cold:
        if test_cold_name != "ULST_COLD":
            tests_data[test_cold_name + "_static"][i, :, :] = (cold_mask_by_test[files[i]][test_cold_name] > 0) & hmask
        else:
            tests_data["ULST" + "_static"][i, :, :] = (tests_data["ULST" + "_static"][i, :, :] > 0) | (
                    (cold_mask_by_test[files[i]][test_cold_name] > 0) & hmask)
    tests_data["ACSPO_mod" + "_static"][i, :, :] = (
            ((tests_data["Adaptive"][i, :, :] > 0) & (~hmask)) | (
            (non_sst_mask > 0) & (tests_data["RGCT"][i, :, :] == 0) & ((tests_data["UNIFORM"][i, :, :] == 0))))
    tests_data["FULL_BTD_MASK"][i, :, :] = (tests_data["FULL_BTD_MASK"][i, :, :] > 0) | (cold_mask & hmask) | (
            tests_data["delta_sst"][i, :, :] < -6.0)
    # adapt_mask = adaptive_boundaries_third_prototype(sst=sst_regression, cmc=sst_reynolds, threshold_min=-0.73,
    #                                                  threshold_max=-0.25, radius=dilatation_size,
    #                                                  threshold_gradient=thresh, window_size=19, scale=3.2,
    #                                                  cold_mask=tests_data["FULL_BTD_MASK"][i, :, :] > 0)
    min_threshold = -1.5
    min_static_mask = (sst_regression - sst_min) < min_threshold
    adapt_mask = (adaptive_test(image=(sst_regression - sst_min), window_size=17, scale=6,
                                threshold=min_threshold, mask=min_static_mask | cold_mask) & (
                              (sst_regression - sst_min) < -0.25)) | min_static_mask
    if switch_adaptive:
        tests_data["FULL_BTD_MASK"][i, :, :] = (tests_data["FULL_BTD_MASK"][i, :, :] > 0) | (min_static_mask & hmask)
    if not switch_full_btds:
        tests_data["FULL_BTD_MASK"][i, :, :] = (tests_data["FULL_BTD_MASK"][i, :, :] > 0) | Adaptive | (
                non_sst_mask > 0)
# gradient_mask_folder = "./gradient_mask/"
# remove_files(gradient_mask_folder)
# draw_a_mask_upon(files, tests_data, high_grad_masks, gradient_mask_folder)


grad_mask_pixel = np.sum(tests_data["GRAD_MASK"])
non_land = np.sum(~np.isnan(tests_data["sst_reynolds"]))
print("Dynamic Areas = %", grad_mask_pixel / non_land * 100.0)
grad_mask_pixel_day = np.sum((tests_data["solzen"] > 0) & tests_data["GRAD_MASK"])

print("Dynamic Areas, day = %", grad_mask_pixel_day / non_land * 100.0)

# save_obj(tests_data, "data")
# path_adaptive = "./only_adaptive_pair/"
# remove_files(path_adaptive)
# output_tiles_adaptive(files=files, tests_data=tests_data, test_names=test_names, output_folder=path_adaptive)
test_names_cold_to_pass = []
for test in test_names_cold:
    if test == "ULST_COLD":
        continue
    test_names_cold_to_pass.append(test)
# "covariance" "RTCT","BT11STD" "BT11_BT8", "BT11_BT4", ,"ULST_COLD", "RFMFT_BT11_BT12"

tests_data["WHOLE"] = np.zeros((68, 256, 256)) < 1.0
tests_data["WARM"] = tests_data["delta_sst"] > 0.0
tests_data["COLD"] = tests_data["delta_sst"] <= 0.0
if switch_full_btds:
    pass_list = test_names + test_names_cold_to_pass
else:
    pass_list = test_names + test_names_cold_to_pass + ["ACSPO_mod"]
rankings_of_btd(tests_data, pass_list,
                key_word="WHOLE")  # , key_word="GRAD_MASK" + ["ACSPO_mod"]
# ranking_individual_btd(tests_data, test_names, files)

sys.exit()
n_tests = -1
if switch_adaptive:
    path_figures = "./figures/"
else:
    path_figures = "./figures_no_adaptive/"
remove_files(path_figures)

if switch_full_btds:
    title_in_figure = "Pure_BTDs\nno ACSPO"
else:
    title_in_figure = "Hybrid BTDs\nand ACSPO"
output_tiles_BTDS(files, tests_data, test_names, thresholds, path_figures, n_tests, nx=30, ny=5 * (n_tests + 1),
                  show_other=False, title_new=title_in_figure)
remove_files("./sst/")
sys.exit()
output_files_sst_BTD(files=files, tests_data=tests_data, output_folder="./sst/")
# output_tiles_Warm(files, tests_data, test_names, thresholds, "./figures_warm/")

sys.exit()
