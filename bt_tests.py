import cv2
import pickle
import numpy as np
import os
import netCDF4 as nc

from MakeRGB import my_cmap
from matplotlib import pyplot as plt
from copy import deepcopy
from water_vapor import get_BTS, dict_BTS_9, dict_BTS_10
from scipy.stats import pearsonr
from adaptive_test import adaptive_test

temperature_flags_list = ["PFMFT_BT11_BT12", "NFMFT_BT11_BT12", "BT11_BT8", "BT12_BT8", "BT11_BT4", "BT12_BT4",
                          "EMISS4", "WATER_VAPOR_TEST", "RTCT", "BT11STD", "BT12_BT4_NIGHT", "BT12_BT4_DAY",
                          "RFMFT_BT11_BT12", "SST_BT12", "SST_BT11", "SST_BT11_COLD", "SST_BT4_COLD", "SST_BT4",
                          "ULST", "ULST_COLD", "WATER_VAPOR_TEST_10", "BT11_BT7"]
shape_type = cv2.MORPH_ELLIPSE

def mask_around(mask, size):
    mask_f = mask.astype(np.float32)
    mas_f_b = cv2.blur(mask_f, (size, size)) > 0
    return mas_f_b


def remove_files(path: str):
    files_to_delete = os.listdir(path)
    for file_d in files_to_delete:
        os.remove(os.path.join(path, file_d))


def get_high_gradient(sst_reynolds, threshold, radius):
    sst_max = np.nanmax(sst_reynolds)
    sst_min = np.nanmin(sst_reynolds)
    sst_to_erode = np.where(np.isnan(sst_reynolds), sst_max, sst_reynolds)
    sst_to_dilate = np.where(np.isnan(sst_reynolds), sst_min, sst_reynolds)
    element_gradient = cv2.getStructuringElement(shape_type, (2 * radius + 1, 2 * radius + 1),
                                                 (radius, radius))
    # (radius, radius) eq. (-1,-1)
    sst_reynolds_max = cv2.dilate(sst_to_dilate, element_gradient)
    sst_reynolds_min = cv2.erode(sst_to_erode, element_gradient)
    return (sst_reynolds_max - sst_reynolds_min) > threshold


# MORPH_RECT

def high_gradient_sst(sst_reynolds, threshold, radius):
    sst_max = np.nanmax(sst_reynolds)
    sst_min = np.nanmin(sst_reynolds)
    sst_to_erode = np.where(np.isnan(sst_reynolds), sst_max, sst_reynolds)

    element_gradient = cv2.getStructuringElement(shape_type, (2 * radius + 1, 2 * radius + 1),
                                                 (radius, radius))

    sst_reynolds_min = cv2.erode(sst_to_erode, element_gradient)
    sst_to_dilate = np.where(np.isnan(sst_reynolds), sst_min, sst_reynolds_min)
    sst_reynolds_max = cv2.dilate(sst_to_dilate, element_gradient)

    return sst_reynolds_max, sst_reynolds_min


def ranking_individual_btd(tests_data: dict, test_names: list, files: list):
    full_test_data = {}
    unique_test_data = {}
    unique_valid_test_data = {}
    true_positive_data = {}
    false_positive_data = {}
    u_metric = {}
    ppv_value = {}
    sensitivity = {}
    u_valid_metric = {}
    sum_true_positives = 0
    sum_false_positives = 0
    f = open("Stats_Data_Tiles.dat", "w")
    fp = open("Stats_Data_Tiles_positives.dat", "w")
    for index in range(len(files)):
        file = files[index]
        BTD_MASK = tests_data["FULL_BTD_MASK"][index, :, :]
        Validation_MASK = tests_data["Validation"][index, :, :]
        Original_MASK = tests_data["Original"][index, :, :]
        legit_mask = ~(np.isnan(tests_data["sst_regression"]) > 0)
        # BTD
        true_negative_total_btd = ((legit_mask > 0) & (~(BTD_MASK > 0))) & (
                (BTD_MASK > 0) == (Validation_MASK > 0))
        false_negative_btd = ((legit_mask > 0) & (~(BTD_MASK > 0))) & (
                (BTD_MASK > 0) != (Validation_MASK > 0))
        true_positives_btd = (BTD_MASK > 0) & (
                (BTD_MASK > 0) == (Validation_MASK > 0))
        false_positives_btd = (BTD_MASK > 0) & (
                (BTD_MASK > 0) != (Validation_MASK > 0))
        # ACSPO
        true_negative_total_acspo = ((legit_mask > 0) & (~(Original_MASK > 0))) & (
                (Original_MASK > 0) == (Validation_MASK > 0))
        false_negative_total_acspo = ((legit_mask > 0) & (~(Original_MASK > 0))) & (
                (Original_MASK > 0) != (Validation_MASK > 0))
        true_positives_acspo = (Original_MASK > 0) & (
                (Original_MASK > 0) == (Validation_MASK > 0))
        false_positives_acspo = (Original_MASK > 0) & (
                (Original_MASK > 0) != (Validation_MASK > 0))
        f.write(file)
        f.write("\n")
        output_to_file = ["true_negative_total_btd", np.sum(true_negative_total_btd), "false_negative_total_btd",
                          np.sum(false_negative_btd)]
        f.writelines(str(output_to_file))
        f.write("\n")
        output_to_file = ["true_negative_total_acspo", np.sum(true_negative_total_acspo), "false_negative_total_acspo",
                          np.sum(false_negative_total_acspo)]
        f.writelines(str(output_to_file))
        f.write("\n")
        f.write("\n")

        fp.write(file)
        fp.write("\n")
        output_to_file = ["true_positives_btd", np.sum(true_positives_btd), "false_positives_btd",
                          np.sum(false_positives_btd)]
        fp.writelines(str(output_to_file))
        fp.write("\n")
        output_to_file = ["true_positives_acspo", np.sum(true_positives_acspo), "false_positives_acspo",
                          np.sum(false_positives_acspo)]
        fp.writelines(str(output_to_file))
        fp.write("\n")
        fp.write("\n")
    f.close()
    fp.close()


def rankings_of_btd(tests_data: dict, test_names: list, key_word="WHOLE"):
    full_test_data = {}
    unique_test_data = {}
    unique_valid_test_data = {}
    true_positive_data = {}
    false_positive_data = {}
    u_metric = {}
    ppv_value = {}
    sensitivity = {}
    u_valid_metric = {}
    sum_true_positives = 0
    sum_false_positives = 0
    for test_name in test_names:
        full_test_data[test_name] = (tests_data[test_name + "_static"] > 0)
    for test_name in test_names:
        # copy all the tests
        test_names_excluded = deepcopy(test_names)
        # remove test name
        test_names_excluded.remove(test_name)
        # initiate an empty mask
        mask = np.zeros(full_test_data[test_name].shape).astype(np.bool)
        mask[:] = False
        # mask from others test
        for test_ex in test_names_excluded:
            mask = mask | full_test_data[test_ex]
        # what test uniquely identified
        unique_test_data[test_name] = (full_test_data[test_name] > 0) & ((full_test_data[test_name] > 0) != (mask > 0))
        unique_valid_test_data[test_name] = (unique_test_data[test_name] > 0) & (tests_data["Validation"] > 0)
        u_valid_metric[test_name] = np.sum(unique_valid_test_data[test_name]) / np.sum(
            unique_test_data[test_name]) * 100.0
        true_positive_data[test_name] = (full_test_data[test_name] > 0) & (
                (full_test_data[test_name] > 0) == (tests_data["Validation"] > 0))
        sum_true_positives = sum_true_positives + np.sum(true_positive_data[test_name])
        false_positive_data[test_name] = (full_test_data[test_name] > 0) & (
                (full_test_data[test_name] > 0) != (tests_data["Validation"] > 0))
        u_metric[test_name] = np.sum(unique_test_data[test_name]) / np.sum(full_test_data[test_name]) * 100.0
        sum_false_positives = sum_false_positives + np.sum(false_positive_data[test_name])
        ppv_value[test_name] = np.sum(true_positive_data[test_name]) / (
                np.sum(false_positive_data[test_name]) + np.sum(true_positive_data[test_name])) * 100.0
        sensitivity[test_name] = np.sum(full_test_data[test_name]) / np.sum(tests_data["FULL_BTD_MASK"]) * 100.0
    f = open("Stats_Data.dat", "w")
    f_time = open("Stats_Data_Day_Night.dat", "w")
    for test_name in test_names:
        print(test_name, "true positive = ", np.sum(true_positive_data[test_name]), "false positive",
              np.sum(false_positive_data[test_name]))
        output_to_file = [test_name, "ppv=", ppv_value[test_name], "u_metric=", u_metric[test_name],
                          "u_valid_metric = ", u_valid_metric[test_name], "sensitivity=",
                          sensitivity[test_name]]
        f.writelines(str(output_to_file))
        f.write("\n")

    legit_mask = ~(np.isnan(tests_data["delta_sst"]) > 0)

    # new mask
    true_positives_total = (tests_data["FULL_BTD_MASK"] > 0) & (
            (tests_data["FULL_BTD_MASK"] > 0) == (tests_data["Validation"] > 0)) & tests_data[key_word]
    false_positives_total = (tests_data["FULL_BTD_MASK"] > 0) & (
            (tests_data["FULL_BTD_MASK"] > 0) != (tests_data["Validation"] > 0)) & tests_data[key_word]
    true_negative_total = ((legit_mask > 0) & (~(tests_data["FULL_BTD_MASK"] > 0))) & (
            (tests_data["FULL_BTD_MASK"] > 0) == (tests_data["Validation"] > 0)) & tests_data[key_word]
    false_negative_total = ((legit_mask > 0) & (~(tests_data["FULL_BTD_MASK"] > 0))) & (
            (tests_data["FULL_BTD_MASK"] > 0) != (tests_data["Validation"] > 0)) & tests_data[key_word]
    ppv_total = (np.sum(true_positives_total)) / (np.sum(true_positives_total) + np.sum(false_positives_total)) * 100.0
    fdr_total = (np.sum(false_positives_total)) / (np.sum(true_positives_total) + np.sum(false_positives_total)) * 100.0
    npv_total = (np.sum(true_negative_total)) / (np.sum(true_negative_total) + np.sum(false_negative_total)) * 100.0
    for_total = (np.sum(false_negative_total)) / (np.sum(true_negative_total) + np.sum(false_negative_total)) * 100.0
    print("BTDs, DCT MASK")
    print("ppv_total", ppv_total, "fdr_total", fdr_total, "npv_total", npv_total, "for_total", for_total)
    output_to_file = str(
        ["ppv_total", ppv_total, "fdr_total", fdr_total, "npv_total", npv_total, "for_total", for_total])
    f.write("\nBTDs, DCT MASK\n")
    f.writelines(str(output_to_file))
    print("CONFUSION BTD:")
    print("true_positives_total", np.sum(true_positives_total))
    print("false_positives_total", np.sum(false_positives_total))
    print("true_negative_total", np.sum(true_negative_total))
    print("false_negative_total", np.sum(false_negative_total))
    TP = np.sum(true_positives_total)
    TN = np.sum(true_negative_total)
    FP = np.sum(false_positives_total)
    FN = np.sum(false_negative_total)
    PSS = TP / (TP + FN) - FP / (FP + TN)
    ACC = (TP + TN) / (TP + FN + FP + TN)
    print("HYBRID ACCURACY = ", ACC)
    print("HYBRID PSS = ", PSS)
    # for day
    true_positives_day = (true_positives_total > 0) & (tests_data["solzen"] > 0)
    false_positives_day = (false_positives_total > 0) & (tests_data["solzen"] > 0)
    true_negative_day = (true_negative_total > 0) & (tests_data["solzen"] > 0)
    false_negative_day = (false_negative_total > 0) & (tests_data["solzen"] > 0)
    if key_word == "WHOLE":
        print("true_positives_day", np.sum(true_positives_day))
        print("false_positives_day", np.sum(false_positives_day))
        print("true_negative_day", np.sum(true_negative_day))
        print("false_negative_day", np.sum(false_negative_day))
    TP = np.sum(true_positives_day)
    TN = np.sum(true_negative_day)
    FP = np.sum(false_positives_day)
    FN = np.sum(false_negative_day)
    PSS = TP / (TP + FN) - FP / (FP + TN)
    ACC = (TP + TN) / (TP + FN + FP + TN)
    if key_word == "WHOLE":
        print("HYBRID ACCURACY DAY = ", ACC)
        print("HYBRID PSS DAY = ", PSS)
    ppv_total = (np.sum(true_positives_day)) / (np.sum(true_positives_day) + np.sum(false_positives_day)) * 100.0
    fdr_total = (np.sum(false_positives_day)) / (np.sum(true_positives_day) + np.sum(false_positives_day)) * 100.0
    npv_total = (np.sum(true_negative_day)) / (np.sum(true_negative_day) + np.sum(false_negative_day)) * 100.0
    for_total = (np.sum(false_negative_day)) / (np.sum(true_negative_day) + np.sum(false_negative_day)) * 100.0
    output_day = ["ppv_total_day", ppv_total, "fdr_total_day", fdr_total, "npv_total_day", npv_total, "for_total_day",
                  for_total]
    print("ppv_total_day", ppv_total, "fdr_total_day", fdr_total, "npv_total_day", npv_total, "for_total_day",
          for_total)
    f_time.write("HYBRID MASK\n")
    f_time.write(str(output_day))
    f_time.write("\n")
    # for night

    true_positives_night = (true_positives_total > 0) & ~(tests_data["solzen"] > 0)
    false_positives_night = (false_positives_total > 0) & ~(tests_data["solzen"] > 0)
    true_negative_night = (true_negative_total > 0) & ~(tests_data["solzen"] > 0)
    false_negative_night = (false_negative_total > 0) & ~(tests_data["solzen"] > 0)
    if key_word == "WHOLE":
        print("true_positives_night", np.sum(true_positives_night))
        print("false_positives_night", np.sum(false_positives_night))
        print("true_negative_night", np.sum(true_negative_night))
        print("false_negative_night", np.sum(false_negative_night))

    TP = np.sum(true_positives_night)
    TN = np.sum(true_negative_night)
    FP = np.sum(false_positives_night)
    FN = np.sum(false_negative_night)
    PSS = TP / (TP + FN) - FP / (FP + TN)
    ACC = (TP + TN) / (TP + FN + FP + TN)
    if key_word == "WHOLE":
        print("HYBRID ACCURACY NIGHT = ", ACC)
        print("HYBRID PSS NIGHT = ", PSS)

    ppv_total = (np.sum(true_positives_night)) / (np.sum(true_positives_night) + np.sum(false_positives_night)) * 100.0
    fdr_total = (np.sum(false_positives_night)) / (np.sum(true_positives_night) + np.sum(false_positives_night)) * 100.0
    npv_total = (np.sum(true_negative_night)) / (np.sum(true_negative_night) + np.sum(false_negative_night)) * 100.0
    for_total = (np.sum(false_negative_night)) / (np.sum(true_negative_night) + np.sum(false_negative_night)) * 100.0
    output_night = ["ppv_total_night", ppv_total, "fdr_total_night", fdr_total, "npv_total_night", npv_total,
                    "for_total_night",
                    for_total]
    print("ppv_total_night", ppv_total, "fdr_total_night", fdr_total, "npv_total_night", npv_total, "for_total_night",
          for_total)
    f_time.write(str(output_night))
    f_time.write("\n\n\n")

    # old mask ACSPO
    # just a correction
    # tests_data["Original"] = (tests_data["Original"] > 0) & (
    #             (tests_data["covariance"] > 0.7) | tests_data["WARM"] | (tests_data["RGCT"] == 0) | (
    #                 tests_data["UNIFORM"] == 0))
    #
    true_positives_total = (tests_data["Original"] > 0) & (
            (tests_data["Original"] > 0) == (tests_data["Validation"] > 0)) & tests_data[key_word]
    false_positives_total = (tests_data["Original"] > 0) & (
            (tests_data["Original"] > 0) != (tests_data["Validation"] > 0)) & tests_data[key_word]
    true_negative_total = ((legit_mask > 0) & (~(tests_data["Original"] > 0))) & (
            (tests_data["Original"] > 0) == (tests_data["Validation"] > 0)) & tests_data[key_word]
    false_negative_total = ((legit_mask > 0) & (~(tests_data["Original"] > 0))) & (
            (tests_data["Original"] > 0) != (tests_data["Validation"] > 0)) & tests_data[key_word]
    ppv_total = (np.sum(true_positives_total)) / (np.sum(true_positives_total) + np.sum(false_positives_total)) * 100.0
    fdr_total = (np.sum(false_positives_total)) / (np.sum(true_positives_total) + np.sum(false_positives_total)) * 100.0
    npv_total = (np.sum(true_negative_total)) / (np.sum(true_negative_total) + np.sum(false_negative_total)) * 100.0
    for_total = (np.sum(false_negative_total)) / (np.sum(true_negative_total) + np.sum(false_negative_total)) * 100.0
    print("OLD ACSPO")
    print("ppv_total", ppv_total, "fdr_total", fdr_total, "npv_total", npv_total, "for_total", for_total)
    output_to_file = str(
        ["ppv_total", ppv_total, "fdr_total", fdr_total, "npv_total", npv_total, "for_total", for_total])
    f.write("\nACSPO MASK\n")
    f.writelines(str(output_to_file))
    print("CONFUSION ACSPO:")
    print("true_positives_total", np.sum(true_positives_total))
    print("false_positives_total", np.sum(false_positives_total))
    print("true_negative_total", np.sum(true_negative_total))
    print("false_negative_total", np.sum(false_negative_total))
    TP = np.sum(true_positives_total)
    TN = np.sum(true_negative_total)
    FP = np.sum(false_positives_total)
    FN = np.sum(false_negative_total)
    PSS = TP / (TP + FN) - FP / (FP + TN)
    ACC = (TP + TN) / (TP + FN + FP + TN)

    print("ACSPO ACCURACY = ", ACC)
    print("ACSPO PSS = ", PSS)
    # for day
    true_positives_day = (true_positives_total > 0) & (tests_data["solzen"] > 0)
    false_positives_day = (false_positives_total > 0) & (tests_data["solzen"] > 0)
    true_negative_day = (true_negative_total > 0) & (tests_data["solzen"] > 0)
    false_negative_day = (false_negative_total > 0) & (tests_data["solzen"] > 0)
    if key_word == "WHOLE":
        print("true_positives_day", np.sum(true_positives_day))
        print("false_positives_day", np.sum(false_positives_day))
        print("true_negative_day", np.sum(true_negative_day))
        print("false_negative_day", np.sum(false_negative_day))

    TP = np.sum(true_positives_day)
    TN = np.sum(true_negative_day)
    FP = np.sum(false_positives_day)
    FN = np.sum(false_negative_day)
    PSS = TP / (TP + FN) - FP / (FP + TN)
    ACC = (TP + TN) / (TP + FN + FP + TN)
    if key_word == "WHOLE":
        print("ACSPO ACCURACY DAY = ", ACC)
        print("ACSPO PSS DAY = ", PSS)

    ppv_total = (np.sum(true_positives_day)) / (np.sum(true_positives_day) + np.sum(false_positives_day)) * 100.0
    fdr_total = (np.sum(false_positives_day)) / (np.sum(true_positives_day) + np.sum(false_positives_day)) * 100.0
    npv_total = (np.sum(true_negative_day)) / (np.sum(true_negative_day) + np.sum(false_negative_day)) * 100.0
    for_total = (np.sum(false_negative_day)) / (np.sum(true_negative_day) + np.sum(false_negative_day)) * 100.0

    print("ppv_total_day", ppv_total, "fdr_total_day", fdr_total, "npv_total_day", npv_total, "for_total_day",
          for_total)
    f_time.write("ACSPO MASK ORIGINAL\n")
    output_day = ["ppv_total_day", ppv_total, "fdr_total_day", fdr_total, "npv_total_day", npv_total, "for_total_day",
                  for_total]
    f_time.write(str(output_day))
    f_time.write("\n")
    # for night

    true_positives_night = (true_positives_total > 0) & ~(tests_data["solzen"] > 0)
    false_positives_night = (false_positives_total > 0) & ~(tests_data["solzen"] > 0)
    true_negative_night = (true_negative_total > 0) & ~(tests_data["solzen"] > 0)
    false_negative_night = (false_negative_total > 0) & ~(tests_data["solzen"] > 0)
    if key_word == "WHOLE":
        print("true_positives_night", np.sum(true_positives_night))
        print("false_positives_night", np.sum(false_positives_night))
        print("true_negative_night", np.sum(true_negative_night))
        print("false_negative_night", np.sum(false_negative_night))

    ppv_total = (np.sum(true_positives_night)) / (np.sum(true_positives_night) + np.sum(false_positives_night)) * 100.0
    fdr_total = (np.sum(false_positives_night)) / (np.sum(true_positives_night) + np.sum(false_positives_night)) * 100.0
    npv_total = (np.sum(true_negative_night)) / (np.sum(true_negative_night) + np.sum(false_negative_night)) * 100.0
    for_total = (np.sum(false_negative_night)) / (np.sum(true_negative_night) + np.sum(false_negative_night)) * 100.0

    TP = np.sum(true_positives_night)
    TN = np.sum(true_negative_night)
    FP = np.sum(false_positives_night)
    FN = np.sum(false_negative_night)
    PSS = TP / (TP + FN) - FP / (FP + TN)
    ACC = (TP + TN) / (TP + FN + FP + TN)
    if key_word == "WHOLE":
        print("ACSPO ACCURACY NIGHT = ", ACC)
        print("ACSPO PSS NIGHT = ", PSS)

    output_night = ["ppv_total_night", ppv_total, "fdr_total_night", fdr_total, "npv_total_night", npv_total,
                    "for_total_night",
                    for_total]
    print("ppv_total_night", ppv_total, "fdr_total_night", fdr_total, "npv_total_night", npv_total, "for_total_night",
          for_total)
    f_time.write(str(output_night))
    f_time.write("\n\n\n")


def pearson(image1: np.array, image2: np.array, rad: int):
    rad = 2 * rad + 1
    image1_blurred = cv2.blur(image1, (rad, rad))
    image2_blurred = cv2.blur(image2, (rad, rad))
    product = (image1 - image1_blurred) * (image2 - image2_blurred)
    product_blurred = cv2.blur(product, (rad, rad))
    correlation = product_blurred
    # print((np.min(image1_square_blurred)))
    return correlation


plank_fk1_7, plank_fk2_7, plank_bc1_7, plank_bc2_7 = 202263.0, 3698.19, 0.43361, 0.99939


def correlation_coeff(bt_1, bt_2, rad):
    rows, cols = bt_1.shape
    coeff = np.empty((rows, cols))
    coeff[:, :] = np.NaN
    for i in range(rad, rows - rad - 1):
        for j in range(rad, cols - rad - 1):
            b1_slice = bt_1[i - rad:i + rad + 1, j - rad:j + rad + 1]
            b2_slice = bt_2[i - rad:i + rad + 1, j - rad:j + rad + 1]
            nan_mask = (~np.isnan(b1_slice)) & (~np.isnan(b2_slice))
            if np.sum(nan_mask) > 1:
                coeff[i, j] = pearsonr(b1_slice[nan_mask], b2_slice[nan_mask])[0]
    return coeff


def Plank_function(T):
    return plank_fk1_7 / np.exp(plank_fk2_7 / (T * plank_bc2_7 + plank_bc1_7) - 1.0)


def harmonics_cuts(image: np.array, size: int):
    rows, cols = image.shape
    Q = np.zeros(image.shape)
    Q[:, :] = np.NaN
    for i in range(size, rows - size):
        for j in range(size, cols - size):
            local_image = image[i - size:i + size, j - size:j + size]
            mean = np.sum(local_image) / 4 / size / size
            if np.isnan(mean):
                continue
            dct_image = cv2.dct(local_image - mean)
            dct_image[size:, :] = 0.0
            dct_image[:, size:] = 0.0
            idct_image = cv2.idct(dct_image) + mean
            diff = local_image - idct_image
            Q[i, j] = np.std(diff)
    return Q


def harmonics_cuts_median(image: np.array, size: int):
    rows, cols = image.shape
    Q = np.zeros(image.shape)
    Q[:, :] = np.NaN
    image_smoothed = deepcopy(image)
    for i in range(size, rows - size):
        for j in range(size, cols - size):
            local_image = image[i - size:i + size, j - size:j + size]
            mean = np.nanmedian(local_image)
            image_smoothed[i, j] = image[i, j] - mean

    for i in range(size, rows - size):
        for j in range(size, cols - size):
            local_image = image_smoothed[i - size:i + size, j - size:j + size]
            mean = np.nanmean(local_image)
            dct_image = cv2.dct(local_image - mean)
            dct_image[size:, :] = 0.0
            dct_image[:, size:] = 0.0
            idct_image = cv2.idct(dct_image) + mean
            diff = local_image - idct_image
            Q[i, j] = np.nanstd(diff)
    return Q


def uniformity_test(image: np.array, size: int):
    ksize = 2 * size + 1
    mean_nan = np.nanmean(image)
    nan_mask = np.isnan(image)
    image[nan_mask] = mean_nan
    median_blur = cv2.medianBlur(image, ksize)
    image_diff = image - median_blur
    image_diff_square = image_diff ** 2
    image_diff_mean = cv2.blur(image_diff, (ksize, ksize))
    image_diff_square_mean = cv2.blur(image_diff_square, (ksize, ksize))
    out = image_diff_square_mean - image_diff_mean ** 2
    out[nan_mask] = np.NaN
    return out


def uniformity_test_slice(image: np.array, size: int, mask):
    rows, cols = image.shape
    Q = np.zeros(image.shape)
    Q[:, :] = np.NaN
    image_to_pass = np.where(mask > 0, np.NaN, image)
    for i in range(size, rows - size - 1):
        for j in range(size, cols - size - 1):
            local_image = image_to_pass[i - size:i + size + 1, j - size:j + size + 1]
            mean = np.nanmedian(local_image)
            diff = local_image - mean
            Q[i, j] = np.nanstd(diff)
    return Q


def save_obj(obj, name):
    with open('obj/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def compute_BTDs_threholds(files: list, path: str, tests_data: dict, test_names: list):
    get_BTS()
    for i in range(len(files)):
        # read the needed data
        file = files[i]
        date = file[29:44]
        local_rows = int(file[62:67])
        local_cols = int(file[69:74])
        data = nc.Dataset(os.path.join(path, file), "r")
        solar_zenith_angle = np.array(data["solar_zenith_angle"][:])
        brightness_temp_ch7 = np.array(data["brightness_temp_ch7"][:])  # 4nm
        brightness_temp_ch11 = np.array(data["brightness_temp_ch11"][:])  # 8nm
        brightness_temp_ch14 = np.array(data["brightness_temp_ch14"][:])  # 11nm
        brightness_temp_ch15 = np.array(data["brightness_temp_ch15"][:])  # 12nm
        individual = np.array(data["individual_clear_sky_tests_results"][:])
        atm_diff = np.array(data.variables["atm_diff"][:])
        sst_regression = np.array(data["sst_regression"][:])
        sst_reynolds = np.array(data["sst_reynolds"][:])
        brightness_temp_crtm_ch7 = np.array(data["brightness_temp_crtm_ch7"][:])
        brightness_temp_crtm_ch11 = np.array(data["brightness_temp_crtm_ch11"][:])
        brightness_temp_crtm_ch14 = np.array(data["brightness_temp_crtm_ch14"][:])
        brightness_temp_crtm_ch15 = np.array(data["brightness_temp_crtm_ch15"][:])
        validation_mask = np.array(data["validation_mask"][:]).astype(np.bool)
        original_mask = np.array(data["original_mask"][:]).astype(np.bool)
        glint = np.array(data["glint"][:]).astype(np.bool)
        data.close()

        # Radiance
        rad_rat_observed = Plank_function(brightness_temp_ch7) / Plank_function(brightness_temp_ch14)
        rad_rat_clear_Sky = Plank_function(brightness_temp_crtm_ch7) / Plank_function(brightness_temp_crtm_ch14)
        solze_mask_night = solar_zenith_angle < 90.0  # day
        # BTDs
        dt_pfmft = brightness_temp_ch14 - brightness_temp_ch15  # 11 nm - 12 nm
        dt_pfmft_rtm = brightness_temp_crtm_ch14 - brightness_temp_crtm_ch15
        # PFMFT_BT11_BT12
        if "PFMFT_BT11_BT12" in test_names:
            tests_data["PFMFT_BT11_BT12"][i, :, :] = dt_pfmft - dt_pfmft_rtm
        # NFMFT_BT11_BT12
        if "NFMFT_BT11_BT12" in test_names:
            tests_data["NFMFT_BT11_BT12"][i, :, :] = dt_pfmft_rtm - dt_pfmft
        # RFMFT
        if "RFMFT_BT11_BT12" in test_names:
            dilatation_size = 7
            element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                                (dilatation_size, dilatation_size))
            brightness_temp_ch14_dilate = cv2.dilate(brightness_temp_ch14, element)
            brightness_temp_ch15_dilate = cv2.dilate(brightness_temp_ch15, element)
            dt_dilate = brightness_temp_ch14_dilate - brightness_temp_ch15_dilate
            tests_data["RFMFT_BT11_BT12"][i, :, :] = np.abs(dt_dilate - dt_pfmft)
        # BT 11 BT 8
        if "BT11_BT8" in test_names:
            dt_bt11_bt_8 = brightness_temp_ch14 - brightness_temp_ch11  # 11 nm - 8 nm
            dt_bt11_bt_8_crtm = brightness_temp_crtm_ch14 - brightness_temp_crtm_ch11
            tests_data["BT11_BT8"][i, :, :] = (dt_bt11_bt_8_crtm - dt_bt11_bt_8)
        # BT 12 BT 8
        if "BT12_BT8" in test_names:
            dt_bt12_bt_8 = brightness_temp_ch15 - brightness_temp_ch11  # 12 nm - 8 nm
            dt_bt12_bt_8_crtm = brightness_temp_crtm_ch15 - brightness_temp_crtm_ch11
            tests_data["BT12_BT8"][i, :, :] = (dt_bt12_bt_8_crtm - dt_bt12_bt_8)
        # BT 11 BT 4
        if "BT11_BT4" in test_names:
            dt_bt11_bt_4 = brightness_temp_ch14 - brightness_temp_ch7  # 11 nm - 4 nm
            dt_bt11_bt_4_crtm = brightness_temp_crtm_ch14 - brightness_temp_crtm_ch7
            dt_bt11_bt_4[solze_mask_night] = np.NaN
            dt_bt11_bt_4_crtm[solze_mask_night] = np.NaN
            tests_data["BT11_BT4"][i, :, :] = (dt_bt11_bt_4_crtm - dt_bt11_bt_4)
        # BT 11 BT 4
        if "BT12_BT4" in test_names:
            dt_bt12_bt_4 = brightness_temp_ch15 - brightness_temp_ch7  # 12 nm - 4 nm
            dt_bt12_bt_4_crtm = brightness_temp_crtm_ch15 - brightness_temp_crtm_ch7
            dt_bt12_bt_4[solze_mask_night] = np.NaN
            dt_bt12_bt_4_crtm[solze_mask_night] = np.NaN
            tests_data["BT12_BT4"][i, :, :] = (dt_bt12_bt_4_crtm - dt_bt12_bt_4)
        # BT12_4_NIGHT
        if "BT12_BT4_NIGHT" in test_names:
            dt_bt12_bt_4 = brightness_temp_ch15 - brightness_temp_ch7  # 12 nm - 4 nm
            dt_bt12_bt_4[solze_mask_night] = np.NaN
            tests_data["BT12_BT4_NIGHT"][i, :, :] = -dt_bt12_bt_4
        # BT12_4_DAY
        if "BT12_BT4_DAY" in test_names:
            dt_bt12_bt_4 = brightness_temp_ch15 - brightness_temp_ch7  # 12 nm - 4 nm
            dt_bt12_bt_4[~solze_mask_night] = np.NaN
            tests_data["BT12_BT4_DAY"][i, :, :] = -dt_bt12_bt_4

        # EMISS4
        if "EMISS4" in test_names:
            emissivity = (rad_rat_observed - rad_rat_clear_Sky) / rad_rat_clear_Sky
            emissivity[glint > 0] = np.NaN
            tests_data["EMISS4"][i, :, :] = emissivity
        # ULST
        if "ULST" in test_names:
            ulst = rad_rat_clear_Sky - rad_rat_observed
            ulst[solze_mask_night] = np.NaN
            tests_data["ULST"][i, :, :] = ulst

        # SST_BT11
        if "SST_BT11" in test_names:
            tests_data["SST_BT11"][i, :, :] = (
                    (sst_regression - sst_reynolds) - (brightness_temp_ch14 - brightness_temp_crtm_ch14))
        # SST_BT12
        if "SST_BT12" in test_names:
            tests_data["SST_BT12"][i, :, :] = (
                    (sst_regression - sst_reynolds) - (brightness_temp_ch15 - brightness_temp_crtm_ch15))
        # SST_BT4
        if "SST_BT4" in test_names:
            dsst_bt4 = (
                    (sst_regression - sst_reynolds) - (brightness_temp_ch7 - brightness_temp_crtm_ch7))
            dsst_bt4[solze_mask_night] = np.NaN
            tests_data["SST_BT4"][i, :, :] = dsst_bt4
        # cold
        # SST_BT11_COLD
        if "SST_BT11_COLD" in test_names:
            tests_data["SST_BT11_COLD"][i, :, :] = -(
                    (sst_regression - sst_reynolds) - (brightness_temp_ch14 - brightness_temp_crtm_ch14))
        # SST_BT11_COLD
        if "SST_BT11_COLD" in test_names:
            tests_data["SST_BT11_COLD"][i, :, :] = -(
                    (sst_regression - sst_reynolds) - (brightness_temp_ch15 - brightness_temp_crtm_ch15))
        # SST_BT4_COLD
        if "SST_BT4_COLD" in test_names:
            dsst_bt4 = (
                    (sst_regression - sst_reynolds) - (brightness_temp_ch7 - brightness_temp_crtm_ch7))
            dsst_bt4[solze_mask_night] = np.NaN
            tests_data["SST_BT4_COLD"][i, :, :] = -dsst_bt4
        # UNI SST
        if "UNI_SST" in test_names:
            kernel_size = 1
            tests_data["UNI_SST"][i, :, :] = uniformity_test(sst_regression, kernel_size)
        # RTCT
        if "RTCT" in test_names:
            radius = 1
            element_gradient = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * radius + 1, 2 * radius + 1),
                                                         (radius, radius))
            brightness_temp_ch14_max = cv2.dilate(brightness_temp_ch14, element_gradient)
            tests_data["RTCT"][i, :, :] = brightness_temp_ch14_max - brightness_temp_ch14
        if "BT11STD" in test_names:
            radius = 1
            ksize = 2 * radius + 1
            image_diff_mean = cv2.blur(brightness_temp_ch14, (ksize, ksize))
            image_diff_square_mean = cv2.blur(brightness_temp_ch14 ** 2, (ksize, ksize))
            tests_data["BT11STD"][i, :, :] = image_diff_square_mean - image_diff_mean ** 2

        # RGCT
        if "RGCT" in test_names:
            tests_data["RGCT"][i, :, :] = individual >> 4 & 1

        # # CROSS_CORR
        # if "CROSS_CORR" in test_names:
        #     tests_data["CROSS_CORR"][i, :, :] = individual >> 7 & 1
        # UNI_EMISS4
        # if "UNI_EMISS4" in test_names:
        #     kernel_size = 1
        #     emissivity = (rad_rat_observed - rad_rat_clear_Sky) / rad_rat_clear_Sky
        #     emissivity[glint > 0] = np.NaN
        #     tests_data["UNI_EMISS4"][i, :, :] = uniformity_test(emissivity, kernel_size)
        # # PEARSON
        # if "PEARSON" in test_names:
        #     kernel_size = 7
        #     correlation = pearson(sst_regression - atm_diff, atm_diff, kernel_size)
        #     correlation[correlation > 1.0] = 1.0
        #     correlation[correlation < -1.0] = -1.0
        #     tests_data["PEARSON"][i, :, :] = correlation
        # # EMISS4_GLINT
        # if "EMISS4_GLINT" in test_names:
        #     dt_bt7_bt7_crtm = brightness_temp_crtm_ch7 - brightness_temp_ch7
        #     dt_bt7_bt7_crtm[glint == 0] = np.NaN
        #     tests_data["EMISS4_GLINT"][i, :, :] = dt_bt7_bt7_crtm
        if "N_OTC" in test_names:
            tests_data["N_OTC"][i, :, :] = brightness_temp_ch7 - brightness_temp_ch15
        if "WATER_VAPOR_TEST" in test_names:
            print(date, local_rows, local_cols)

            bt_ch9 = dict_BTS_9[date][local_rows:local_rows + 256, local_cols:local_cols + 256]
            tests_data["WATER_VAPOR_TEST"][i, :, :] = pearson(bt_ch9, brightness_temp_ch14, 2)
        if "WATER_VAPOR_TEST_10" in test_names:
            print(date, local_rows, local_cols)

            bt_ch10 = dict_BTS_10[date][local_rows:local_rows + 256, local_cols:local_cols + 256]
            std_threshold = 0.25
            ksize = 1
            image_diff_mean = cv2.blur(brightness_temp_ch14, (ksize, ksize))
            image_diff_square_mean = cv2.blur(brightness_temp_ch14 ** 2, (ksize, ksize))
            mask_bt14 = image_diff_square_mean - image_diff_mean ** 2 > std_threshold
            image_diff_mean = cv2.blur(bt_ch10, (ksize, ksize))
            image_diff_square_mean = cv2.blur(bt_ch10 ** 2, (ksize, ksize))
            mask_bt10 = image_diff_square_mean - image_diff_mean ** 2 > std_threshold
            mask_total = mask_bt10 & mask_bt14
            cross_corr = pearson(bt_ch10, brightness_temp_ch14, ksize)
            cross_corr[mask_total] = np.NaN
            tests_data["WATER_VAPOR_TEST_10"][i, :, :] = cross_corr
        if "BT11_BT7" in test_names:
            print(date, local_rows, local_cols)
            bt_ch10 = dict_BTS_10[date][local_rows:local_rows + 256, local_cols:local_cols + 256]
            tests_data["BT11_BT7"][i, :, :] = bt_ch10 - brightness_temp_ch14
        # validation mask
        tests_data["Validation"][i, :, :] = validation_mask
        # individual mask
        static_mask = individual >> 2 & 1
        adaptive_mask = individual >> 3 & 1
        rgct_mask = individual >> 4 & 1
        uniformity_mask = individual >> 6 & 1
        cross_corr_mask = individual >> 7 & 1
        pure_adaptive = adaptive_mask != static_mask
        # static mask
        tests_data["Static"][i, :, :] = static_mask
        # adaptive mask
        tests_data["Adaptive"][i, :, :] = adaptive_mask
        # Individual mask
        tests_data["Individual"][i, :, :] = static_mask | adaptive_mask | rgct_mask | uniformity_mask | cross_corr_mask
        # original mask
        tests_data["Original"][i, :, :] = original_mask
        # delta sst
        tests_data["delta_sst"][i, :, :] = (sst_regression - sst_reynolds)
        # solzen
        tests_data["solzen"][i, :, :] = solze_mask_night
        # sst_reynolds
        tests_data["sst_reynolds"][i, :, :] = sst_reynolds
        # sst_regression
        tests_data["sst_regression"][i, :, :] = sst_regression
        # not_sst_mask
        tests_data["non_sst_mask"][i, :, :] = rgct_mask | uniformity_mask | cross_corr_mask
    return tests_data


def output_files_sst_BTD(files, tests_data, output_folder):
    font_size = 26
    label_size = 20
    map_cloud = deepcopy(my_cmap)
    map_cloud.set_over((128 / 255.0, 128 / 255.0, 128 / 255.0))
    for i in range(len(files)):
        file = files[i]
        print(file)
        output_path = output_folder + file[0:74] + ".png"
        fig = plt.figure(figsize=(30, 10))
        mask_BTD = tests_data["FULL_BTD_MASK"][i, :, :]
        Original = tests_data["Original"][i, :, :]
        Validation = tests_data["Validation"][i, :, :]
        nan_mask = np.isnan(tests_data["sst_regression"][i, :, :])
        sst_regression = tests_data["sst_regression"][i, :, :]
        sst_reynolds = tests_data["sst_reynolds"][i, :, :]
        max_val_sst = np.nanmax(sst_regression)
        min_val_sst = np.nanmin(sst_regression)
        max_val_cmc = np.nanmax(sst_reynolds)
        min_val_cmc = np.nanmin(sst_reynolds)
        delta_sst = tests_data["delta_sst"][i, :, :]
        delta_sst[delta_sst > 2.0] = 2.0
        delta_sst[delta_sst < -2.0] = -2.0
        max_val_to_show = max_val_sst
        min_val_to_show = max(min_val_sst, min_val_cmc - 2.0)
        # plotting the BTDs mask
        fig.add_subplot(1, 3, 1)
        to_show = np.where(mask_BTD > 0, 100.0 + max_val_to_show, sst_regression)
        to_show[nan_mask > 0] = np.NaN

        plt.imshow(to_show, interpolation="none", vmin=min_val_to_show, vmax=max_val_to_show, cmap=map_cloud)
        plt.title("Hybrid BTDs and ACSPO mask", fontsize=font_size)
        cbar3 = plt.colorbar(fraction=0.046, pad=0.04)
        cbar3.ax.tick_params(labelsize=label_size)
        plt.xticks([])
        plt.yticks([])
        # plotting the Validation mask
        fig.add_subplot(1, 3, 2)
        to_show = np.where(Validation > 0, 100.0 + max_val_to_show, sst_regression)
        to_show[nan_mask > 0] = np.NaN
        plt.imshow(to_show, interpolation="none", vmin=min_val_to_show, vmax=max_val_to_show, cmap=map_cloud)
        plt.title("Validation Mask", fontsize=font_size)
        cbar3 = plt.colorbar(fraction=0.046, pad=0.04)
        cbar3.ax.tick_params(labelsize=label_size)
        plt.xticks([])
        plt.yticks([])
        # plotting the Original mask
        fig.add_subplot(1, 3, 3)
        to_show = np.where(Original > 0, 100.0 + max_val_to_show, sst_regression)
        to_show[nan_mask > 0] = np.NaN
        plt.imshow(to_show, interpolation="none", vmin=min_val_to_show, vmax=max_val_to_show, cmap=map_cloud)
        plt.title("Original ACSPO Mask", fontsize=font_size)
        cbar3 = plt.colorbar(fraction=0.046, pad=0.04)
        cbar3.ax.tick_params(labelsize=label_size)
        plt.xticks([])
        plt.yticks([])
        fig.savefig(output_path, transparent=True, format="png")
        plt.close(fig)


def output_tiles_BTDS(files, tests_data, test_names, thresholds, output_folder, n_tests=-1, nx=30, ny=60,
                      show_other=True, title_new="Beta 2.9", title_old="ACSPO 2.8"):
    n_figures = len(test_names) + 1
    if n_tests > 0:
        n_figures = n_tests + 1
    else:
        n_figures = 1
    n_figures = 1
    n_col = 6
    if ~show_other:
        n_col = 3
    font_size = 26
    label_size = 20
    map_cloud = deepcopy(my_cmap)
    map_cloud.set_over((128 / 255.0, 128 / 255.0, 128 / 255.0))
    for i in range(len(files)):
        file = files[i]
        print(file)
        output_path = output_folder + file[0:74] + ".png"
        fig = plt.figure(figsize=(30, 10))
        mask_BTD = tests_data["FULL_BTD_MASK"][i, :, :]
        Original = tests_data["Original"][i, :, :]
        Validation = tests_data["Validation"][i, :, :]
        nan_mask = np.isnan(tests_data["sst_regression"][i, :, :])
        delta_sst = tests_data["delta_sst"][i, :, :]
        delta_sst[delta_sst > 2.0] = 1.9999
        delta_sst[delta_sst < -2.0] = -1.9999
        # plotting the BTDs mask
        fig.add_subplot(1, 3, 1)
        to_show = np.where(mask_BTD > 0, 100.0, delta_sst)
        to_show[nan_mask > 0] = np.NaN
        plt.imshow(to_show, interpolation="none", vmin=-2, vmax=2, cmap=map_cloud)
        plt.title(title_new, fontsize=font_size)
        cbar3 = plt.colorbar(fraction=0.046, pad=0.04)
        cbar3.ax.tick_params(labelsize=label_size)
        plt.xticks([])
        plt.yticks([])
        # plotting the Validation mask
        fig.add_subplot(1, 3, 2)
        to_show = np.where(Validation > 0, 100.0, delta_sst)
        to_show[nan_mask > 0] = np.NaN
        plt.imshow(to_show, interpolation="none", vmin=-2, vmax=2, cmap=map_cloud)
        plt.title("Validation Mask", fontsize=font_size)
        cbar3 = plt.colorbar(fraction=0.046, pad=0.04)
        cbar3.ax.tick_params(labelsize=label_size)
        plt.xticks([])
        plt.yticks([])
        # plotting the Original mask
        fig.add_subplot(1, 3, 3)
        to_show = np.where(Original > 0, 100.0, delta_sst)
        to_show[nan_mask > 0] = np.NaN
        plt.imshow(to_show, interpolation="none", vmin=-2, vmax=2, cmap=map_cloud)
        plt.title(title_old, fontsize=font_size)
        cbar3 = plt.colorbar(fraction=0.046, pad=0.04)
        cbar3.ax.tick_params(labelsize=label_size)
        plt.xticks([])
        plt.yticks([])
        # if show_other:
        #     # plotting the Static Original mask
        #     fig.add_subplot(n_figures, n_col, 4)
        #     to_show = np.where(delta_sst < -1.8, 100.0, delta_sst)
        #     to_show[nan_mask] = np.NaN
        #     plt.imshow(to_show, interpolation="none", vmin=-2, vmax=2, cmap=map_cloud)
        #     plt.title("Static Test,\n original threshold = -1.8 K", fontsize=font_size)
        #     cbar3 = plt.colorbar(fraction=0.046, pad=0.04)
        #     cbar3.ax.tick_params(labelsize=label_size)
        #     plt.xticks([])
        #     plt.yticks([])
        #     # plotting the Adaptive Original mask
        #     fig.add_subplot(n_figures, n_col, 5)
        #     mask_adaptive_static = (tests_data["Adaptive"][i, :, :] > 0) | (tests_data["Static"][i, :, :] > 0)
        #     to_show = np.where(mask_adaptive_static, 100.0, delta_sst)
        #     to_show[nan_mask] = np.NaN
        #     plt.imshow(to_show, interpolation="none", vmin=-2, vmax=2, cmap=map_cloud)
        #     plt.title("Static and Adaptive Test,Original", fontsize=font_size)
        #     cbar3 = plt.colorbar(fraction=0.046, pad=0.04)
        #     cbar3.ax.tick_params(labelsize=label_size)
        #     plt.xticks([])
        #     plt.yticks([])
        #     # plotting the Static  mask, Different Threshold
        #     fig.add_subplot(n_figures, n_col, 6)
        #     to_show = np.where(delta_sst < -5, 100.0, delta_sst)
        #     to_show[nan_mask] = np.NaN
        #     plt.imshow(to_show, interpolation="none", vmin=-2, vmax=2, cmap=map_cloud)
        #     plt.title("Static Test,\n  threshold = -5 K", fontsize=font_size)
        #     cbar3 = plt.colorbar(fraction=0.046, pad=0.04)
        #     cbar3.ax.tick_params(labelsize=label_size)
        #     plt.xticks([])
        #     plt.yticks([])
        #
        # i_fig = n_col + 1
        # for test_name in test_names:
        #     if i_fig > n_figures * n_col:
        #         continue
        #     fig.add_subplot(n_figures, n_col, i_fig)
        #     adaptive_mask = tests_data[test_name + "_adaptive"][i, :, :] > 0
        #     static_mask = tests_data[test_name + "_static"][i, :, :] > 0
        #     image = tests_data[test_name][i, :, :]
        #     image[np.isnan(image)] = 0.0
        #     threshold = thresholds[test_name]
        #     to_show = np.where(static_mask | adaptive_mask, 100.0, image)
        #     to_show[nan_mask] = np.NaN
        #     plt.imshow(to_show, interpolation="none", vmin=0, vmax=threshold, cmap=map_cloud)
        #     plt.title(test_name, fontsize=font_size)
        #     cbar3 = plt.colorbar(fraction=0.046, pad=0.04)
        #     cbar3.ax.tick_params(labelsize=label_size)
        #     plt.xticks([])
        #     plt.yticks([])
        #     i_fig += n_col

        fig.savefig(fname=output_path, transparent=True, format='png')
        plt.close(fig)


def output_tiles_adaptive(files, tests_data, test_names, output_folder):
    n_figures = len(test_names)
    n_col = 2
    font_size = 10
    label_size = 10
    map_cloud = deepcopy(my_cmap)
    map_cloud.set_over((128 / 255.0, 128 / 255.0, 128 / 255.0))
    for i in range(len(files)):
        file = files[i]
        print(file)
        output_path = output_folder + file + ".jpg"
        fig = plt.figure(figsize=(10, 5 * (len(test_names) + 1)))
        i_fig = 1
        for test_name in test_names:
            adaptive_mask = tests_data[test_name + "_adaptive"][i, :, :]
            static_mask = tests_data[test_name + "_static"][i, :, :]
            nan_mask = np.isnan(tests_data["delta_sst"][i, :, :])
            delta_sst = tests_data["delta_sst"][i, :, :]
            delta_sst[delta_sst > 2.0] = 2.0
            delta_sst[delta_sst < -2.0] = -2.0
            to_show = np.where((static_mask > 0), 100.0, delta_sst)
            to_show[nan_mask] = np.NaN
            fig.add_subplot(n_figures, n_col, i_fig)
            plt.imshow(to_show, interpolation="none", vmin=-2, vmax=2, cmap=map_cloud)
            plt.title("Static_{}".format(test_name), fontsize=font_size)
            cbar3 = plt.colorbar(fraction=0.046, pad=0.04)
            cbar3.ax.tick_params(labelsize=label_size)
            plt.xticks([])
            plt.yticks([])
            # add adaptive
            i_fig = i_fig + 1
            fig.add_subplot(n_figures, n_col, i_fig)
            to_show_2 = np.where((adaptive_mask > 0) | (static_mask > 0), 100.0, delta_sst)
            print("adaptive_mask_impact", np.sum((adaptive_mask > 0)), np.sum(static_mask))
            to_show_2[nan_mask] = np.NaN
            fig.add_subplot(n_figures, n_col, i_fig)
            plt.imshow(to_show_2, interpolation="none", vmin=-2, vmax=2, cmap=map_cloud)
            plt.title("Adaptive_{}".format(test_name), fontsize=font_size)
            cbar3 = plt.colorbar(fraction=0.046, pad=0.04)
            cbar3.ax.tick_params(labelsize=label_size)
            plt.xticks([])
            plt.yticks([])
            i_fig = i_fig + 1

        plt.savefig(output_path)
        plt.close(fig)


def draw_a_mask_upon(files, tests_data, masks, output_folder):
    map_cloud = deepcopy(my_cmap)
    map_cloud.set_over((128 / 255.0, 128 / 255.0, 128 / 255.0))
    for i in range(len(files)):
        file = files[i]
        output_path = output_folder + file + ".jpg"
        delta_sst = tests_data["delta_sst"][i, :, :]
        sst_reynolds = tests_data["sst_reynolds"][i, :, :]
        mask = ~masks[file]

        nan_mask = np.isnan(tests_data["sst_regression"][i, :, :])

        fig = plt.figure(figsize=(20, 10))

        fig.add_subplot(1, 3, 1)
        to_show = np.where((mask > 0), 100.0, delta_sst)
        to_show[nan_mask] = np.NaN
        plt.imshow(to_show, interpolation="none", vmin=-2, vmax=2, cmap=map_cloud)
        plt.title("BTD_MASK", fontsize=15)
        cbar3 = plt.colorbar(fraction=0.046, pad=0.04)
        cbar3.ax.tick_params(labelsize=15)
        plt.xticks([])
        plt.yticks([])

        fig.add_subplot(1, 3, 2)
        to_show2 = np.where((mask > 0), np.nanmax(sst_reynolds) * 5, sst_reynolds)
        to_show2[nan_mask] = np.NaN
        plt.imshow(to_show2, interpolation="none", vmax=np.nanmax(sst_reynolds), cmap=map_cloud)
        plt.title("BTD_MASK", fontsize=15)
        cbar3 = plt.colorbar(fraction=0.046, pad=0.04)
        cbar3.ax.tick_params(labelsize=15)
        plt.xticks([])
        plt.yticks([])

        fig.add_subplot(1, 3, 3)
        to_show3 = sst_reynolds
        to_show3[nan_mask] = np.NaN
        plt.imshow(to_show3, interpolation="none", cmap=map_cloud)
        plt.title("BTD_MASK", fontsize=15)
        cbar3 = plt.colorbar(fraction=0.046, pad=0.04)
        cbar3.ax.tick_params(labelsize=15)
        plt.xticks([])
        plt.yticks([])

        plt.savefig(output_path)
        plt.close(fig)


def thresholds_cold():
    thresholds = {}
    thresholds2 = {}
    thresholds["PFMFT_BT11_BT12"] = 0.3  # 0.76
    thresholds["NFMFT_BT11_BT12"] = 0.6  # 0.6

    thresholds["RFMFT_BT11_BT12"] = 0.7

    thresholds["RTCT"] = 1.4  # 1.4
    thresholds["BT11STD"] = 0.65  # .65
    thresholds["BT11_BT8"] = 2.1
    thresholds["BT12_BT8"] = 1.1  # 1.3
    thresholds["BT11_BT4"] = 4.0
    thresholds["BT12_BT4"] = 3.0  # 3.5
    thresholds["SST_BT11_COLD"] = 1.6
    thresholds["SST_BT4_COLD"] = 0.4

    #  covariance  thresholds
    thresholds["covariance"] = 0.05

    thresholds["covariance_1"] = 0.07  # 0.07
    thresholds["covariance_2"] = 0.005  # 0.02
    thresholds["covariance_3"] = 0.005  # 0.004
    thresholds["covariance_4"] = 0.005  # 0.0015
    # threshold
    thresholds["covariance_diff_1"] = 0.03
    thresholds["covariance_diff_2"] = 0.01
    thresholds["covariance_diff_3"] = 0.001
    thresholds["covariance_diff_4"] = 0.0004

    thresholds2["SST_BT11_COLD"] = -5.1
    thresholds2["BT11_BT8"] = -1.2
    thresholds2["BT12_BT8"] = -1.2  # -1.3
    thresholds2["BT11_BT4"] = -3.7
    thresholds2["BT12_BT4"] = -3.5  # -4.0

    thresholds["SST_BT11"] = 4.5  # 1.3 # 3.7ye
    thresholds["SST_BT12"] = 4.5  # 1.3 # 3.7

    thresholds["BT11_BT7"] = -55
    thresholds["ULST_COLD"] = 0.10  # was 0.06 # second version 0.12 # 0.10
    thresholds["UNI_SST"] = 0.05
    thresholds["EMISS4"] = 0.15
    thresholds["RGCT"] = 0.0

    # not effective tests
    thresholds["WATER_VAPOR_TEST"] = 0.05
    thresholds["WATER_VAPOR_TEST_10"] = 0.02
    thresholds["N_OTC"] = 16.0
    # thin cloud tests
    thresholds["BT12_BT4_DAY"] = 16.25
    thresholds["BT12_BT4_NIGHT"] = 11.7
    adaptive_radius = {}
    adaptive_clear = {}
    for key in thresholds:
        adaptive_radius[key] = -5
        adaptive_clear[key] = 6
    # adaptive_radius["ULST_COLD"] = 15
    # adaptive_clear["SST_BT11_COLD"] = 10
    return thresholds, thresholds2, adaptive_radius, adaptive_clear


def get_cold_sst_matrix(tests_data, files, test_names, test_dt_threshold):
    thresholds, thresholds2, adaptive_radius, adaptive_clear = thresholds_cold()
    test_for_second_threshold = ["BT11_BT8", "BT12_BT8", "BT11_BT4", "BT12_BT4"]
    cold_masks = {}
    cold_mask_by_test = {}
    for i in range(len(files)):
        file = files[i]
        total_mask = np.zeros((256, 256)) > 1.0
        cold_mask_by_test[file] = {}
        for test_name in test_names:
            threshold = thresholds[test_name]
            if test_name != "ULST_COLD":
                image = tests_data[test_name][i, :, :]
            else:
                image = tests_data["ULST"][i, :, :]
            rad = adaptive_radius[test_name]
            scale = adaptive_clear[test_name]
            mask = (image > threshold) & (~np.isnan(image))
            if test_name in temperature_flags_list:
                mask = (mask & (tests_data["delta_sst"][i, :, :] <= -0.25))
                adapt_mask = adaptive_test(image=image, window_size=rad, scale=scale,
                                           threshold=threshold, mask=mask)
                mask = mask | adapt_mask
                if test_name in test_for_second_threshold:
                    threshold2 = thresholds2[test_name]
                    mask2 = (image < threshold2) & (~np.isnan(image)) & (tests_data["delta_sst"][i, :, :] <= 0)
                    adapt_mask2 = adaptive_test(image=image, window_size=rad, scale=scale,
                                                threshold=threshold2, mask=mask2)
                    mask2 = mask2 | adapt_mask2
                    mask = mask2 | mask  # need to separate later
            total_mask = total_mask | mask
            cold_mask_by_test[file][test_name] = mask

        cold_masks[file] = total_mask

    return cold_masks, cold_mask_by_test


def print_locations(files: list, path: str, output_file_name: str):
    lat_lon = {}
    for file in files:
        path_file = os.path.join(path, file)
        nc_data = nc.Dataset(path_file, mode="r")
        lat = np.array(nc_data["latitude"][:])
        lon = np.array(nc_data["longitude"][:])
        max_lat = np.nanmax(lat)
        min_lat = np.nanmin(lat)
        max_lon = np.nanmax(lon)
        min_lon = np.nanmin(lon)
        lat_lon[file] = max_lat, min_lat, max_lon, min_lon
    f = open(output_file_name, mode="w")
    for key in lat_lon:
        f.write(key)
        f.write("\t=")
        max_lat, min_lat, max_lon, min_lon = lat_lon[key]
        output = str([max_lat, min_lat, max_lon, min_lon])
        f.write(output)
        f.write("\n")
    f.close()
