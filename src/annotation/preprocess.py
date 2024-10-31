"""
Implementation of preprocesses used to extract features from homogeneous data images.
"""

import cv2
import numpy as np

from omegaconf import DictConfig


def minmax(x, span=(0, 1)):
    """
    Minmax transform.

    Args:
        x (np.array): array to transform
        span (list, optional): range to transform to. Defaults to [0,1].

    Returns:
        np.array: scaled array
    """
    std = (x - x.min()) / (x.max() - x.min())
    scaled = std * (span[1] - span[0]) + span[0]
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
        radius: Can be increased to improve expansion speed while causing decreased isolation from nearby areas.
    
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


def morphological_transform(img, ksize=(3,3)):
    """
    Morphological transform used to extract foreground markers from bubble image.

    Args:
        img: image to process
        ksize: kernel size
    Returns:
        np.array: transformed image
    """
    img_open = morphological_top_hat(img)
    img_closed = morphological_bottom_hat(img)

    img_0 = img - img_open + img_closed

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize)

    g_0 = cv2.morphologyEx(img_0, cv2.MORPH_OPEN, kernel)
    f_0c = imreconstruct(img_0, g_0)

    g_0c = cv2.morphologyEx(f_0c, cv2.MORPH_CLOSE, kernel)
    f_0cr = imreconstruct(f_0c, g_0c)

    return f_0cr


def preprocess(img: np.array, cfg: DictConfig):
    """
    Apply preprocessing steps to a grayscale image.

    Args:
        img (np.array): grayscale image
        cfg (DictConfig): configuration for preprocessing
    Returns:
        np.array: processed image
    """
    ssr_img = single_scale_retinex(
        img,
        cfg.single_scale_retinex.sigma,
    ).astype('float32')

    bf = bilateral_filtering(
        ssr_img,
        cfg.bilateral_filtering.diameter,
        cfg.bilateral_filtering.sigma_color,
        cfg.bilateral_filtering.sigma_space,
    )
    morphed = morphological_transform(
        bf,
        cfg.morphological_transform.kernel_size,
    )
    morphed = cv2.GaussianBlur(
        morphed,
        cfg.gaussian_blur.kernel_size,
        cfg.gaussian_blur.sigma_x,
    )

    m = minmax(
        morphed.ravel(),
        cfg.minmax.span
    ).astype('uint8')

    i = m.reshape(img.shape)
    return i
