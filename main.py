import cv2
import os
import numpy as np
import netCDF4 as nc
from matplotlib import pyplot as plt
import matplotlib
import sys

cdict = {'red': ((0.0, 0.0, 0.0),
                 (0.1, 0.5, 0.5),
                 (0.2, 0.0, 0.0),
                 (0.4, 0.2, 0.2),
                 (0.6, 0.0, 0.0),
                 (0.8, 1.0, 1.0),
                 (1.0, 1.0, 1.0)),
         'green': ((0.0, 0.0, 0.0),
                   (0.1, 0.0, 0.0),
                   (0.2, 0.0, 0.0),
                   (0.4, 1.0, 1.0),
                   (0.6, 1.0, 1.0),
                   (0.8, 1.0, 1.0),
                   (1.0, 0.0, 0.0)),
         'blue': ((0.0, 0.0, 0.0),
                  (0.1, 0.5, 0.5),
                  (0.2, 1.0, 1.0),
                  (0.4, 1.0, 1.0),
                  (0.6, 0.0, 0.0),
                  (0.8, 0.0, 0.0),
                  (1.0, 0.0, 0.0))}

my_cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap', cdict, 256)

my_cmap.set_bad("saddlebrown")
my_cmap.set_over("gray")
my_cmap.set_under("gray")
path: str = r"C:/Users/Alexander Semenov/PycharmProjects/untitled/images/"
out_path: str = r"C:/Users/Alexander Semenov/PycharmProjects/untitled/masked_images/"

path_to = r"D:\Users\Alexander\ACSPO\Opencv\temps"

files = os.listdir(path)

file = files[0]


# mark holes
def mark(img):
    # crop margins
    # margin = 4
    # img = img[margin:-margin, margin:-margin]
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # threshold
    # gray = cv2.GaussianBlur(gray, (7, 7), 0)
    _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # mask = cv2.bitwise_not(mask)

    # redraw
    # img[mask == 0] = (100, 100, 0)  # 0 - black, no intesity, 255 - full intesity , white
    return mask


path_nc = r"C:\Users\Alexander Semenov\Desktop\ABI_ACSPO_CODE\cloudmasktest\tiles_pics\all_areas"
path_mask = r"C:\\Users\\Alexander Semenov\\Desktop\\ABI_ACSPO_CODE\\cloudmasktest\\tiles_pics\\binaries\\"

path_out = r"C:\\Users\Alexander Semenov\Desktop\ABI_ACSPO_CODE\cloudmasktest\tiles_pics\sst_otsu\\"


nc_files_list = os.listdir(path_nc)
png_files_list = os.listdir(path_mask)
test_files_list = os.listdir(path_to)
png_to_nc_dict: dict = {}

png_to_check : dict = {}

for nc_file_name in nc_files_list:
    for pnh_file in png_files_list:
        if nc_file_name in pnh_file:
            png_to_nc_dict[nc_file_name] = pnh_file

n_figures = 4
for nc_file_name in nc_files_list:
    for pnh_file in test_files_list:
        if nc_file_name in pnh_file:
            png_to_check[nc_file_name] = pnh_file



for name_nc in nc_files_list:
    name_mask = png_to_nc_dict[name_nc]
    name_to_check2 = png_to_check[name_nc]
    full_pat_nc = os.path.join(path_nc, name_nc)

    full_path_mask = os.path.join(path_mask, name_mask)
    path_tp_pass = os.path.join(path_to, name_to_check2)

    file_nc = nc.Dataset(full_pat_nc, "r")

    delta_t = np.array(file_nc.variables["sst_regression"][:]) - np.array(file_nc.variables["sst_reynolds"][:])
    original_mask = np.array(file_nc.variables["original_mask"][:])
    validation_mask = np.array(file_nc.variables["validation_mask"][:])
    individual = np.array(file_nc.variables["individual_clear_sky_tests_results"][:])
    extra = np.array(file_nc.variables["extra_byte_clear_sky_tests_results"][:])

    static_mask = individual >> 2 & 1 | (extra >> 0 & 1)
    adaptive_mask = individual >> 3 & 1 | (extra >> 1 & 1)
    pure_adaptive = adaptive_mask != static_mask
    rgct_mask = individual >> 4 & 1
    uniformity_mask = individual >> 6 & 1
    cross_corr_mask = individual >> 7 & 1
    pos_mask = extra >> 3 & 1
    total_mask = static_mask | adaptive_mask | rgct_mask | uniformity_mask | cross_corr_mask | pos_mask
    rest_mask = rgct_mask | uniformity_mask | cross_corr_mask | pos_mask


    upper_limit = np.nanmax(np.array(file_nc.variables["sst_reynolds"][:])) + 2.0
    lower_limit = np.nanmin(np.array(file_nc.variables["sst_reynolds"][:])) - 2.0

    sst_regression = np.array(file_nc.variables["sst_regression"][:])
    sst_reynolds = np.array(file_nc.variables["sst_reynolds"][:])

    sst_regression = np.where(sst_regression > upper_limit, upper_limit, sst_regression)
    sst_regression = np.where(sst_regression < lower_limit, lower_limit, sst_regression)
    T_fill = 321

    delta_t = np.where(delta_t > 1.8, 1.8, delta_t)
    delta_t = np.where(delta_t < -1.8, -1.8, delta_t)



    img = cv2.imread(path_tp_pass)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(img, (3, 3), 0)

    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    laplacian_2 = np.abs(np.where(delta_t < -0.5, laplacian, 0))
    laplace_mask = (laplacian_2 > 5.0)

    print(laplace_mask)

    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    both_sobel = cv2.bitwise_or(sobelx, sobely)

    mask = np.array(cv2.imread(full_path_mask, cv2.IMREAD_GRAYSCALE))

    delta_t_masked = np.where((mask == 0), delta_t, 2.0)
    # delta_t_masked[laplace_mask] = 2.0
    delta_t_masked = np.where(np.isnan(delta_t), np.NaN, delta_t_masked)
    delta_t_masked_original = np.where(original_mask == 0, delta_t, 2.0)
    delta_t_masked_original = np.where(np.isnan(delta_t), np.NaN, delta_t_masked_original)
    delta_t_masked_validation = np.where(validation_mask == 0, delta_t, 2.0)
    delta_t_masked_validation = np.where(np.isnan(delta_t), np.NaN, delta_t_masked_validation)



    sst_masked = np.where((mask== 0), sst_regression,T_fill)
    # sst_masked[laplace_mask] = T_fill
    sst_masked[rest_mask] = T_fill
    sst_masked = np.where(np.isnan(sst_regression), np.NaN, sst_masked)
    sst_regression_original = np.where(original_mask == 0, sst_regression,T_fill)
    sst_regression_original = np.where(pure_adaptive == 0, sst_regression_original, lower_limit-2)
    sst_regression_original = np.where(np.isnan(sst_regression), np.NaN, sst_regression_original)

    sst_regression_validation = np.where(validation_mask == 0, sst_regression,T_fill)
    sst_regression_validation = np.where(np.isnan(sst_regression), np.NaN, sst_regression_validation)










    fig = plt.figure(figsize=(60, 20))
    plt.rcParams.update({'font.size': 30})
    fig.add_subplot(1, n_figures, 1)
    plt.imshow(sst_masked, interpolation="none", cmap=my_cmap, vmin=lower_limit, vmax=upper_limit)
    plt.title("New mask ", fontsize=45)
    cbar1 = plt.colorbar(fraction=0.046, pad=0.04)
    cbar1.ax.tick_params(labelsize=30)
    plt.xticks([])
    plt.yticks([])

    fig.add_subplot(1, n_figures, 2)
    plt.imshow(sst_regression_original, interpolation="none", cmap=my_cmap, vmin=lower_limit, vmax=upper_limit)
    plt.title("Old MASK ", fontsize=45)
    cbar2 = plt.colorbar(fraction=0.046, pad=0.04)
    cbar2.ax.tick_params(labelsize=30)
    plt.xticks([])
    plt.yticks([])

    fig.add_subplot(1, n_figures, 4)
    plt.imshow(sst_regression_validation, interpolation="none", cmap=my_cmap, vmin=lower_limit, vmax=upper_limit)
    plt.title("Validation Mask", fontsize=45)
    cbar3 = plt.colorbar(fraction=0.046, pad=0.04)
    cbar3.ax.tick_params(labelsize=30)
    plt.xticks([])
    plt.yticks([])

    fig.add_subplot(1, n_figures, 3)
    plt.imshow(sst_regression, interpolation="none", cmap=my_cmap, vmin=lower_limit, vmax=upper_limit)
    plt.title("No Masking", fontsize=45)
    cbar4 = plt.colorbar(fraction=0.046, pad=0.04)
    cbar4.ax.tick_params(labelsize=30)
    plt.xticks([])
    plt.yticks([])

    plt.savefig(path_out + "//sst_data_{}.jpg".format(name_nc))
    plt.close(fig)


sys.exit()
#
# src = cv2.imread(path_tp_pass, cv2.IMREAD_UNCHANGED)
# src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
#
# # apply guassian blur on src image
# dst = cv2.GaussianBlur(src, (5, 5), cv2.BORDER_DEFAULT)
#
# # display input and output image
# cv2.imshow("Gaussian Smoothing", np.hstack((src, dst)))
# cv2.waitKey(0)  # waits until a key is pressed
#
# fig = plt.figure(figsize=(20, 10))
# fig.add_subplot(1, 2, 1)
# plt.hist(src.ravel(), 256, [0, 256])
# fig.add_subplot(1, 2, 2)
# plt.hist(dst.ravel(), 256, [0, 256])
# plt.show()
#
# mask1 = mark(src)
# mask2 = mark(dst)
# cv2.imshow("Gaussian Smoothing 2", np.hstack((mask1, mask2)))
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()  # destroys the window showing image
