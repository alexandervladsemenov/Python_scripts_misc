import cv2
import numpy as np


def blur_over_mask(array, mask, size: int):
    ksize = (size, size)
    image_to_feed = np.where(mask, array, 0.0)
    mask_to_average = mask.astype(np.float)
    blurred_image = cv2.blur(image_to_feed, ksize)
    mask_blurred = cv2.blur(mask_to_average, ksize)
    return np.where(mask_blurred >= 1.0 / size, blurred_image / mask_blurred, np.NaN)


def adaptive_test(image: np.array, window_size: int, threshold, scale: float, mask, *args, **kwargs):
    if window_size <= 0:
        return np.zeros(image.shape) > 1.0
    image_square = np.power(image, 2)
    image_blurred = blur_over_mask(image, mask, window_size)
    image_blurred_square = blur_over_mask(image_square, mask, window_size)
    sigma = image_blurred_square - np.power(image_blurred, 2)
    dt = image - image_blurred
    return (dt ** 2 < sigma * scale / (threshold ** 2) * pow(image, 2)) & ~mask  # / threshold * (image ** 2)
