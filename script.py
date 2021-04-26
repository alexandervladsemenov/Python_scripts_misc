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
from bt_tests import harmonics_cuts, uniformity_test, save_obj, load_obj, compute_BTDs_threholds, output_tiles_Warm, \
    output_tiles_BTDS, rankings_of_btd,remove_files


# def Plank_function(T, wave=3.9):
#     c2 = 1.4387752 * 1e4
#     return 1.0 / (np.exp(c2 / T / wave) - 1.0)


def get_threshold(btd, validation, xmax, title, xmin=0):
    path_to_save = r"D:\Users\Alexander\ACSPO\Opencv\Thresholds"
    validation_mask = validation > 0
    plt_label_x = "BTD,K"
    plt_label_y = "Probability Density per K"
    list_of_radiance_tests = ["EMISS4_GLINT", "EMISS4", "ULST"]
    if title in list_of_radiance_tests:
        plt_label_x = "Relative_Radiance"
        plt_label_y = "Probability Density per Relative_Radiance"
    if title == "SST_DCT":
        plt_label_x = "SST_DCT,K"
    if title == "RGCT":
        return
    if (title == "WATER_VAPOR_TEST_10") | (title == "WATER_VAPOR_TEST_10"):
        plt_label_x = "Pearson_Corr"
        plt_label_y = "Probability Density"
    if (title == "PEARSON"):
        plt_label_x = "Cross_corr"
        plt_label_y = "Probability Density"
        xmin = -1.0
    if (title in ["BT11_BT4", "BT12_BT4"]):
        xmin = 0
    if( title in ["BT11_BT8", "BT12_BT8"]):
        xmin = 0
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

test_names = ["PFMFT_BT11_BT12", "NFMFT_BT11_BT12", "BT11_BT8", "BT12_BT8", "BT11_BT4", "BT12_BT4",
              "EMISS4", "SST_DCT", "UNI_SST", "EMISS4_GLINT", "RGCT",
              # "RFMFT_BT11_BT12",
              "ULST"]

print("number of tests",len(test_names))
# tests_data: dict = {}
tests_data = load_obj("data")
add_names = []
for test_name in add_names:
    tests_data[test_name] = np.empty((len(files), 256, 256))

# validation mask

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

files.sort()

# dilatation_size = 9
# element = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
#                                     (dilatation_size, dilatation_size))


tests_data = compute_BTDs_threholds(files=files, path=path, tests_data=tests_data, test_names=add_names)




save_obj(tests_data, "data")
rankings_of_btd(tests_data,test_names)
sys.exit()


x_range = {}

x_range["PFMFT_BT11_BT12"] = 5.0
x_range["NFMFT_BT11_BT12"] = 2.0
x_range["RFMFT_BT11_BT12"] = 3.0
x_range["BT11_BT8"] = 4
x_range["BT12_BT8"] = 4
x_range["BT11_BT4"] = 7
x_range["BT12_BT4"] = 7
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
# save_obj(tests_data, "data")
# for test_name in test_names:
#     print(test_name)
#     get_threshold(tests_data[test_name], tests_data["Validation"], xmax=x_range[test_name], title=test_name)  #
#
# sys.exit()

thresholds = {}
thresholds["PFMFT_BT11_BT12"] = 1
thresholds["NFMFT_BT11_BT12"] = 0.5
thresholds["RFMFT_BT11_BT12"] = 0.7
thresholds["BT11_BT8"] = 1.3
thresholds["BT12_BT8"] = 1.3
thresholds["BT11_BT4"] = 4.0
thresholds["BT12_BT4"] = 4.0
thresholds["BT12_BT4_NIGHT"] = 12
thresholds["BT12_BT4_DAY"] = 16
thresholds["SST_BT11"] = 4.5  # 1.3 # 3.7ye
thresholds["SST_BT12"] = 4.5  # 1.3 # 3.7
thresholds["SST_DCT"] = 0.3
thresholds["ULST"] = 0.05
thresholds["ULST2"] = 0.05
thresholds["UNI_SST"] = 0.05
thresholds["UNI_SST2"] = 0.05
thresholds["UNI_ULST"] = 0.01
thresholds["UNI_EMISS4"] = 0.007
thresholds["EMISS4"] = 0.15
thresholds["EMISS4_GLINT"] = 4.1
thresholds["RGCT"] = 0.0
thresholds["CROSS_CORR"] = 0.0
thresholds["WATER_VAPOR_TEST"] = 1.1
thresholds["WATER_VAPOR_TEST_10"] = 1.1
thresholds["LAPLACE"] = 25
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
adaptive_radius["UNI_SST"] = -5
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
adaptive_scale["ULST"] = 6
adaptive_scale["UNI_ULST"] = 0
adaptive_scale["UNI_SST"] = 0
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
        tests_data[test_name + "_static"][i, :, :] = (image > threshold) & (~np.isnan(threshold))
        tests_data[test_name + "_adaptive"][i, :, :] = adaptive_test(image=image, window_size=radius, scale=scale,
                                                                     threshold=threshold) & (~np.isnan(threshold))
        tests_data["FULL_BTD_MASK"][i, :, :] = tests_data["FULL_BTD_MASK"][i, :, :] | (
                tests_data[test_name + "_adaptive"][i, :, :] > 0) | (
                                                       tests_data[test_name + "_static"][i, :, :] > 0)

save_obj(tests_data, "data")
# sys.exit()
n_tests = 1
path_figures =  "./figures/"
remove_files(path_figures)
output_tiles_BTDS(files, tests_data, test_names, thresholds,path_figures, n_tests, nx=20, ny=10,show_other=False)
# output_tiles_Warm(files, tests_data, test_names, thresholds, "./figures_warm/")

sys.exit()
