"""
Implementation of methods used to extract markers from bubble images.
"""

import sys
sys.path.append('..')

import numpy as np
import cv2
from features.preprocessing import *


def _get_markers(img: np.array):
    """
    The threshold is set for a preprocessed image and is not meant to be changed.

    Args:
        img (np.array): processed image
    
    Returns:
        np.array: binary image of thresholded markers
    """
    _, all = cv2.threshold(img, 60, 255, cv2.THRESH_BINARY)
    return all


def _get_big_markers(markers):
    """
    Extract big bubble markers by morphological operations.

    Args:
        markers (np.array): binary mask of all markers

    Returns:
        np.array: big markers
    """
    kernel_b = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    big = cv2.erode(markers, kernel_b, iterations=3)
    big = cv2.dilate(big, kernel_b, iterations=2)
    return big 


def _get_small_markers(markers):
    """
    Extract small bubbles as difference between all bubble markers and big markers.

    Args:
        markers (np.array): binary mask of all markers

    Returns:
        np.array: small bubles mask
    """
    big = cv2.dilate(_get_big_markers(markers),
                     kernel = np.ones((5,5), np.uint8),
                     iterations=6)
    inv_big = 255 - big
    return np.where(inv_big == 255, markers, 0)


def get_markers(marker_type: str, img: np.array):
    """
    Extract bubble markers by threshold.

    Args:
        marker_type (str): 'all', 'big', 'small'
        img (np.array): grayscale image to threshold. should be preprocessed
    
    Returns:
        np.array: binary image of markers
    """

    markers = _get_markers(img)

    if marker_type=='all':
        return markers
    elif marker_type=='big':
        return _get_big_markers(markers)
    elif marker_type=='small':
        return _get_small_markers(markers)