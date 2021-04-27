import cv2
import os
import numpy as np
import netCDF4 as nc
from matplotlib import pyplot as plt
import sys
import seaborn as sns
from scipy.stats import gaussian_kde
from MakeRGB import my_cmap
from scipy import stats
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
from adaptive_test import adaptive_test
from bt_tests import harmonics_cuts, uniformity_test, save_obj, load_obj, compute_BTDs_threholds, output_tiles_adaptive, \
    output_tiles_BTDS, rankings_of_btd, remove_files, get_high_gradient, temperature_flags_list,ranking_individual_btd


# def Plank_function(T, wave=3.9):
#     c2 = 1.4387752 * 1e4
#     return 1.0 / (np.exp(c2 / T / wave) - 1.0)


def get_thresold_hist(btd, validation, xmax, title, additional_mask, xmin=0,
                      path=r"D:\Users\Alexander\ACSPO\Opencv\Thresholds"):
    path_to_save = path
    validation_mask = validation > 0
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
    nb = 200
    max_val = xmax
    array_bins = np.arange(xmin, max_val + max_val / nb, max_val / nb)
    plt.hist(btd[validation_mask & ~np.isnan(btd) & additional_mask], color="blue", bins=array_bins, label='Cloudy')
    plt.hist(btd[~validation_mask & ~np.isnan(btd) & additional_mask], color="red", bins=array_bins, label='Clear')
    plt.legend()
    plt.title("Test = {} clear/cloudy".format(title))
    plt.ylabel(plt_label_y)
    plt.xlabel(plt_label_x)
    plt.savefig(os.path.join(path_to_save, title))
    plt.close()


def get_threshold(btd, validation, xmax, title, xmin=0):
    path_to_save = r"D:\Users\Alexander\ACSPO\Opencv\Thresholds"
    validation_mask = validation > 0
    plt_label_x = "BTD,K"
    plt_label_y = "Probability Density per K"
    list_of_radiance_tests = ["EMISS4_GLINT", "EMISS4", "ULST"]
    if title in list_of_radiance_tests:
        plt_label_x = "Relative_Radiance"
        plt_label_y = "Probability Density per Relative_Radiance"
    if title == "RGCT":
        return
    if (title in ["BT11_BT4", "BT12_BT4"]):
        xmin = -7
    if (title in ["BT11_BT8", "BT12_BT8"]):
        xmin = -3
    #
    # print(len(btd[validation_mask & ~np.isnan(btd)]))
    # print(len(btd[~validation_mask & ~np.isnan(btd)]))
    density_12_8_cloudy = gaussian_kde(btd[validation_mask & ~np.isnan(btd)])

    density_12_8_clear = gaussian_kde(btd[~validation_mask & ~np.isnan(btd)])

    xs = np.linspace(xmin, xmax, 200)
    density_12_8_cloudy.covariance_factor = lambda: .25
    density_12_8_cloudy._compute_covariance()

    density_12_8_clear.covariance_factor = lambda: .25
    density_12_8_clear._compute_covariance()

    plt.plot(xs, density_12_8_cloudy(xs), label="Cloudy")
    plt.plot(xs, density_12_8_clear(xs), label="Clear")
    plt.legend()
    plt.title("Test = {} clear/cloudy".format(title))
    plt.ylabel(plt_label_y)
    plt.xlabel(plt_label_x)
    plt.savefig(os.path.join(path_to_save, title))
    plt.close()


path = r"C:\Users\Alexander Semenov\Desktop\ABI_ACSPO_CODE\cloudmasktest\tiles_pics\all_areas"

files = os.listdir(path)

test_names_original = ["PFMFT_BT11_BT12", "NFMFT_BT11_BT12", "BT11_BT8", "BT12_BT8", "BT11_BT4", "BT12_BT4",
                       "EMISS4", "SST_DCT", "UNI_SST", "EMISS4_GLINT", "RGCT",
                       "RFMFT_BT11_BT12",
                       "ULST"]
test_names = ["UNI_SST", "RGCT","PFMFT_BT11_BT12","BT11_BT8","ULST","SST_BT11"]

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



# add_names = ["UNI_SST", "RGCT", "PFMFT_BT11_BT12", "NFMFT_BT11_BT12", "BT11_BT8", "BT12_BT8", "BT11_BT4", "BT12_BT4",
#              "EMISS4", "EMISS4_GLINT",
#              "RFMFT_BT11_BT12", "SST_BT12", "SST_BT11",
#              "ULST"]

add_names = []
for test_name in add_names:
    tests_data[test_name] = np.empty((len(files), 256, 256))

files.sort()

# tests_data = compute_BTDs_threholds(files=files, path=path, tests_data=tests_data, test_names=add_names)
#
# save_obj(tests_data, "data")
# #
# sys.exit()


x_range = {}

x_range["PFMFT_BT11_BT12"] = 2.0
x_range["NFMFT_BT11_BT12"] = 1.0
x_range["RFMFT_BT11_BT12"] = 1.5
x_range["BT11_BT8"] = 2
x_range["BT12_BT8"] = 2
x_range["BT11_BT4"] = 4
x_range["BT12_BT4"] = 4
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
x_range["WATER_VAPOR_TEST"] = 1.9
x_range["WATER_VAPOR_TEST_10"] = 1.9
x_range["LAPLACE"] = 30
x_range["PEARSON"] = 1.0
save_obj(tests_data, "data")
# for test_name in temperature_flags_list:
#     print(test_name)
#     additional_mask = tests_data["delta_sst"] > 0
#     get_thresold_hist(tests_data[test_name], tests_data["Validation"], xmax=x_range[test_name], title=test_name,
#                       additional_mask=additional_mask, path="./Thresh_warm/")  #
# for test_name in temperature_flags_list:
#     print(test_name)
#     additional_mask = tests_data["delta_sst"] <= 0
#     get_thresold_hist(tests_data[test_name], tests_data["Validation"], xmax=x_range[test_name], title=test_name,
#                       additional_mask=additional_mask, path="./Thresh_cold")  #
# sys.exit()

thresholds = {}
thresholds2 = {}
test_for_second_threshold = ["BT11_BT8", "BT12_BT8", "BT11_BT4", "BT12_BT4"]
thresholds["PFMFT_BT11_BT12"] = 0.8
thresholds["NFMFT_BT11_BT12"] = 0.5
thresholds["RFMFT_BT11_BT12"] = 0.7

thresholds["BT11_BT8"] = 1.6
thresholds["BT12_BT8"] = 1.3
thresholds["BT11_BT4"] = 3.0
thresholds["BT12_BT4"] = 4.0

thresholds2["BT11_BT8"] = -0.9
thresholds2["BT12_BT8"] = -1.3
thresholds2["BT11_BT4"] = -2.5
thresholds2["BT12_BT4"] = -4.0

thresholds["SST_BT11"] = 4.5  # 1.3 # 3.7ye
thresholds["SST_BT12"] = 4.5  # 1.3 # 3.7

thresholds["ULST"] = 0.06
thresholds["UNI_SST"] = 0.05
thresholds["EMISS4"] = 0.15
thresholds["RGCT"] = 0.0

adaptive_radius = {}
adaptive_radius["PFMFT_BT11_BT12"] = 20
adaptive_radius["NFMFT_BT11_BT12"] = 20
adaptive_radius["RFMFT_BT11_BT12"] = 20
adaptive_radius["BT11_BT8"] = 20
adaptive_radius["BT12_BT8"] = 20
adaptive_radius["BT12_BT4_NIGHT"] = 20
adaptive_radius["BT12_BT4_DAY"] = 20
adaptive_radius["BT11_BT4"] = 20
adaptive_radius["BT12_BT4"] = 20
adaptive_radius["SST_BT11"] = 20
adaptive_radius["SST_DCT"] = -5
adaptive_radius["UNI_SST"] = -5  # 10
adaptive_radius["UNI_SST2"] = -5
adaptive_radius["UNI_ULST"] = -5
adaptive_radius["UNI_EMISS4"] = -5
adaptive_radius["SST_BT12"] = 20
adaptive_radius["ULST"] = 20
adaptive_radius["EMISS4"] = 20
adaptive_radius["EMISS4_GLINT"] = 20
adaptive_radius["RGCT"] = -5
adaptive_radius["CROSS_CORR"] = -5
adaptive_radius["WATER_VAPOR_TEST"] = -5
adaptive_radius["WATER_VAPOR_TEST_10"] = -5
adaptive_radius["LAPLACE"] = -5
adaptive_scale = {}

adaptive_scale["PFMFT_BT11_BT12"] = 6
adaptive_scale["NFMFT_BT11_BT12"] = 6
adaptive_scale["RFMFT_BT11_BT12"] = 6
adaptive_scale["BT11_BT8"] = 6
adaptive_scale["BT12_BT8"] = 6
adaptive_scale["BT12_BT4_NIGHT"] = 6
adaptive_scale["BT12_BT4_DAY"] = 6
adaptive_scale["BT11_BT4"] = 6
adaptive_scale["BT12_BT4"] = 6
adaptive_scale["SST_BT11"] = 6
adaptive_scale["SST_BT12"] = 6
adaptive_scale["SST_DCT"] = 6
adaptive_scale["ULST"] = 0
adaptive_scale["UNI_ULST"] = 0
adaptive_scale["UNI_SST"] = 3
adaptive_scale["UNI_SST2"] = 0
adaptive_scale["UNI_EMISS4"] = 0
adaptive_scale["EMISS4"] = 6
adaptive_scale["EMISS4_GLINT"] = 6
adaptive_scale["RGCT"] = 0
adaptive_scale["CROSS_CORR"] = 0
adaptive_scale["WATER_VAPOR_TEST"] = 0
adaptive_scale["WATER_VAPOR_TEST_10"] = 0
adaptive_scale["LAPLACE"] = -5
tests_data["FULL_BTD_MASK"] = np.zeros((68, 256, 256)) > 1.0

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
            mask = (mask & (tests_data["delta_sst"][i, :, :] > 0))
            if test_name in test_for_second_threshold:
                threshold2 = thresholds2[test_name]
                mask2 = (image < threshold2) & (~np.isnan(image)) & (tests_data["delta_sst"][i, :, :] > 0)
                mask = mask2 | mask # need to separate later
        tests_data[test_name + "_static"][i, :, :] = mask

        # disable adaptive test
        radius = -5
        tests_data[test_name + "_adaptive"][i, :, :] = adaptive_test(image=image, window_size=radius, scale=scale,
                                                                     threshold=threshold, mask=mask) & (
                                                           ~np.isnan(image))
        tests_data["FULL_BTD_MASK"][i, :, :] = tests_data["FULL_BTD_MASK"][i, :, :] | (
                tests_data[test_name + "_adaptive"][i, :, :] > 0) | (
                                                       tests_data[test_name + "_static"][i, :, :] > 0)

thresh = 2.0
dilatation_size = 11

for i in range(len(files)):
    individual_original = tests_data["Individual"][i, :, :]
    sst_reynolds = tests_data["sst_reynolds"][i, :, :]
    high_grad_mask = get_high_gradient(sst_reynolds, thresh, dilatation_size)
    tests_data["FULL_BTD_MASK"][i, :, :] = (tests_data["FULL_BTD_MASK"][i, :, :] > 0) | (individual_original > 0)

# save_obj(tests_data, "data")
# path_adaptive = "./only_adaptive_pair/"
# remove_files(path_adaptive)
# output_tiles_adaptive(files=files, tests_data=tests_data, test_names=test_names, output_folder=path_adaptive)

rankings_of_btd(tests_data, test_names)
ranking_individual_btd(tests_data, test_names,files)
# sys.exit()
n_tests = 1
path_figures = "./figures_no_adaptive/"
remove_files(path_figures)
output_tiles_BTDS(files, tests_data, test_names, thresholds, path_figures, n_tests, nx=30, ny=5 * (n_tests + 1),
                  show_other=False)
# output_tiles_Warm(files, tests_data, test_names, thresholds, "./figures_warm/")

sys.exit()
