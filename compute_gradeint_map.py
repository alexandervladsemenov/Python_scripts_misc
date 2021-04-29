import cv2
import os
import numpy as np


def compute_gradient_map(image, rad, threshold):
    grad_x = cv2.Sobel(image.astype(np.float32), cv2.CV_32FC1, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image.astype(np.float32), cv2.CV_32FC1, 0, 1, ksize=3)
    abs_grad = np.abs(grad_x ** 2 + grad_y ** 2)
    mask : np.array = abs_grad > threshold
    mask_float = mask.astype(np.float32)
    element_gradient = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rad + 1, 2 * rad + 1),
                                                 (rad, rad))
    mask_broaden = cv2.dilate(mask_float, element_gradient)
    return mask_broaden > .0
