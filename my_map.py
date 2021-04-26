import matplotlib
import numpy as np
import cv2
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


# my_cmap.set_over("gray")
# my_cmap.set_under("gray")

def normalize(array):
    min_value = np.nanmin(array)
    max_value = np.nanmax(array)
    return (array - min_value) / (max_value - min_value)

def nan_helper(y):

    return np.isnan(y), lambda z: z.nonzero()[0]


def paint(array):
    nan_mask = np.array(np.isnan(array), dtype=np.uint8)
    return  cv2.inpaint(array, nan_mask, 2, cv2.INPAINT_TELEA)