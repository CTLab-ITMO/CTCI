"""
Implementation of preprocesses used to extract features from homogeneous data images.
"""

import cv2
import numpy as np


def minmax(X, range=[0,1]):
    """
    Minmax transform.

    Args:
        X (np.array): array to transform
        range (list, optional): range to transform to. Defaults to [0,1].

    Returns:
        np.array: scaled array
    """
    std = (X - X.min()) / (X.max() - X.min())
    scaled = std * (range[1] - range[0]) + range[0]
    return scaled.astype('uint8')


def single_scale_retinex(img, sigma):
    """
    Single scale retinex algorithm.

    Args:
        img: image to process
        sigma: weight of processing. empirically selected

    Returns:
        np.array: grayscale image with applied retinex
    """
    img = np.where(img > 0, img, 10e-6)
    g_img = cv2.GaussianBlur(img, (0,0), sigma)
    return np.log(img) - np.log(g_img)


def bilateral_filtering(img, diameter, sigma_color, sigma_space):
    """

    Wrapper for opencv bilateral filtering function.

    Args:
        img (np.array): image for processing
        diameter: parameter for cv2.bilateralFilter function
        sigma_color: parameter for cv2.bilateralFilter function
        sigma_space: parameter for cv2.bilateralFilter function

    Returns:
        np.array: transformed image
    """
    return cv2.bilateralFilter(img, diameter, sigma_color, sigma_space)


def morphological_top_hat(gray_img, kernel_size=(3,3)):
    """
    Wrapper for morphological top hat operation.

    Args:
        gray_img: grayscale image
        kernel_size: size of filtering kernel
    
    Returns:
        np.array: transformed image
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    return cv2.morphologyEx(gray_img, cv2.MORPH_TOPHAT, kernel)


def morphological_bottom_hat(gray_img, kernel_size=(3,3)):
    """
    Wrapper for morphological bottom hat operation.

    Args:
        gray_img: grayscale image
        kernel_size: size of filtering kernel
    
    Returns:
        np.array: transformed image
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    return cv2.morphologyEx(gray_img, cv2.MORPH_BLACKHAT, kernel)


def imreconstruct(marker: np.ndarray, mask: np.ndarray, radius: int = 1):
    """
    Iteratively expand the markers white keeping them limited by the mask during each iteration.

    Args:
        marker: Grayscale image where initial seed is white on black background.
        mask: Grayscale mask where the valid area is white on black background.
        radius Can be increased to improve expansion speed while causing decreased isolation from nearby areas. 
    
    Returns:
        A copy of the last expansion.

    Written By Semnodime.
    """
    kernel = np.ones(shape=(radius * 2 + 1,) * 2, dtype=np.uint8)
    while True:
        expanded = cv2.dilate(src=marker, kernel=kernel)
        cv2.bitwise_and(src1=expanded, src2=mask, dst=expanded)

        # Termination criterion: Expansion didn't change the image at all
        if (marker == expanded).all():
            return expanded
        marker = expanded


def morphological_transform(img):
    """
    Morphological transform used to extract foreground markers from bubble image.

    Args:
        img: image to process
    
    Returns:
        np.array: transformed image
    """
    img_open = morphological_top_hat(img)
    img_closed = morphological_bottom_hat(img)

    img_0 = img - img_open + img_closed

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

    g_0 = cv2.morphologyEx(img_0, cv2.MORPH_OPEN, kernel)
    f_0c = imreconstruct(img_0, g_0)

    g_0c = cv2.morphologyEx(f_0c, cv2.MORPH_CLOSE, kernel)
    f_0cr = imreconstruct(f_0c, g_0c)

    return f_0cr


def preprocess(img: np.array):
    """
    Apply preprocessing steps to a grayscale image.

    Args:
        img (np.array): grayscale image
    
    Returns:
        np.array: processed image
    """
    ssr_img = single_scale_retinex(img, 80).astype('float32')
    bf = bilateral_filtering(ssr_img, 5, 75, 75)
    morphed = morphological_transform(bf)
    morphed = cv2.GaussianBlur(morphed, (5,5), 0)
    m = minmax(morphed.ravel(), [0, 255]).astype('uint8')
    i = m.reshape(img.shape)
    return i
