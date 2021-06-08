import sys
from matplotlib import pyplot as plt
import numpy as np
from numpy import random

import cv2
from MakeRGB import my_cmap
from bt_tests import load_obj, save_obj


# two points covariance 1 = 0.3 covariance = 0.0875


def plot_hist(xmax, xmin, nbins, **kwargs):
    maxValue = xmax - xmin
    range_bins = np.arange(xmin, xmax + maxValue / nbins, maxValue / nbins)
    for key in kwargs:
        print(key)
        data = kwargs[key]
        mask = data["mask"]
        var = data["var"]
        title = data["title"]
        color = (random.rand(), random.rand(), random.rand())
        plt.hist(var[mask & ~np.isnan(var)], color=color, bins=range_bins, label=title, alpha=0.5)
        plt.legend()
    plt.show()
    while not plt.waitforbuttonpress(): pass


def mask_right_triangle(covar1, covar2, point1=0.3, point2=0.0875, point_x_1=0.7):
    k = -point1 / point2
    # mask_cloud = covar1 > (covar2 - point_x_1) * k
    return covar1 - covar2 * k


def covariance_2_point_mask(covar1, covar2, point1=None, point2=None):
    if point1 is None:
        point1 = [0.2, 0.005]
    if point2 is None:
        point2 = [0.032, 0.030]
    point3 = [0, 0]
    # line 1
    k1 = (point1[0] - point3[0]) / (point1[1] - point3[1])
    ref1_mask = (covar2 - point3[1]) * k1 + point3[0] > covar1
    # line 2
    k2 = (point2[0] - point3[0]) / (point2[1] - point3[1])
    ref2_mask = (covar2 - point3[1]) * k2 + point3[0] < covar1
    # line 3
    k3 = (point2[0] - point1[0]) / (point2[1] - point1[1])
    ref3_mask = (covar2 - point1[1]) * k3 + point1[0] > covar1
    clear_full_mask = ref1_mask & ref2_mask & ref3_mask
    clear_full_mask = (covar1 < 0.3) & (covar2 < 0.05)
    return ~clear_full_mask & ~np.isnan(covar1)


def plot_two_d_hist(var1, var2, mask, name1="var1", name2="var2", title="histogram", range_hist=None, maxvalue=300,
                    bin_num=40):
    if range_hist is None:
        range_hist = [[0, 3], [0, 10]]
    mask_to_plot = ~np.isnan(var1) & ~np.isnan(var2) & mask
    x_data = var1[mask_to_plot]
    y_data = var2[mask_to_plot]
    plt.hist2d(x_data, y_data, bins=(bin_num, bin_num), vmax=maxvalue, cmap="jet", range=range_hist)
    plt.xlabel(name1)
    plt.ylabel(name2)
    plt.title(title)
    plt.colorbar()
    plt.show()
    while not plt.waitforbuttonpress(): pass
    plt.close()
    return


if __name__ == "__main__":

    data = load_obj(name="covariance")
    only_bts = data["only_bts"]
    sst_bts = data["sst_bts"]
    validation_mask = data["validation_mask"]
    no_mask = data["no_mask"]

    index_bts = 3  # <=3
    index_sst = index_bts
    i = 6

    diff_flag = False

    # plot_two_d_hist(sst_bts[:, :, :, index_sst], sst_bts[:, :, :, index_sst - 1], mask=validation_mask & no_mask[:, :, :],
    #                 name1="covar2",
    #                 name2="covar1", title="clouds", range_hist=[[0, 0.2], [0, 0.4]], maxvalue=40, bin_num=50)
    #
    # plot_two_d_hist(sst_bts[:, :, :, index_sst], sst_bts[:, :, :, index_sst - 1], mask=~validation_mask & no_mask[:, :, :],
    #                 name1="covar2",
    #                 name2="covar1", title="clear", range_hist=[[0, 0.2], [0, 0.4]], maxvalue=40, bin_num=50)


    threshold_value = mask_right_triangle(covar1=sst_bts[:, :, :, index_sst - 1], covar2=sst_bts[:, :, :, index_sst])

    data1 = {"mask": validation_mask & no_mask[:, :, :], "var": threshold_value, "title": "cloudy"}
    data2 = {"mask": ~validation_mask & no_mask[:, :, :], "var": threshold_value, "title": "clear"}

    plot_hist(xmax=0.5,xmin=0,nbins=300,data_clouds = data1, data_clear_sky = data2)


    # tests_data = load_obj("data")
    # tests_data["covariance"] = threshold_value
    # save_obj(tests_data, "data")
    # sys.exit()
    sys.exit()

    if diff_flag:
        if index_bts >= index_sst:
            diff = sst_bts[:, :, :, index_sst] - only_bts[:, :, :, index_bts]
        else:
            diff = only_bts[:, :, :, index_bts] - sst_bts[:, :, :, index_sst]
    else:
        diff = sst_bts[:, :, :, index_sst]
    diff[~no_mask[:, :, :]] = np.NaN

    vmax_range = 0.01 / 5 ** (index_bts - 1)

    if index_bts == 4:
        vmax_range = 0.01 / 5 ** (index_bts - 2)

    plt.imshow(diff[i, :, :], vmax=vmax_range, interpolation="none", cmap=my_cmap)
    plt.colorbar()
    plt.show()
    plt.waitforbuttonpress()
    plt.close()

    nb = 200
    max_val = 0.1 / 5.0 ** (index_bts - 1)
    if index_bts == 4:
        max_val = 0.1 / 5.0 ** (index_bts - 2)
    min_val = 0
    array_bins = np.arange(min_val, max_val + max_val / nb, max_val / nb)

    mask_nan = ~np.isnan(diff)

    plt.hist(diff[validation_mask & mask_nan], color="blue", bins=array_bins, label='Cloudy', alpha=0.5)  # ,alpha= 0.5
    plt.hist(diff[~validation_mask & mask_nan], color="red", bins=array_bins, label='Clear', alpha=0.5)
    plt.legend()
    plt.show()
    #
    plt.waitforbuttonpress()
    #
    tests_data = load_obj("data")
    if not diff_flag:
        tests_data["covariance" + "_" + str(index_bts)] = diff
    else:
        tests_data["covariance" + "_diff_" + str(index_bts)] = diff
    save_obj(tests_data, "data")
    sys.exit()
    # limit 0.005
