"""
Utilities for watershed.
"""
import numpy as np
import cv2


def find_contours(img: np.ndarray, labels: np.ndarray, markers: np.ndarray) -> np.ndarray:
    """
    draw a binary mask of contours between watershed labels

    Args:
        img (np.ndarray): grayscale image
        labels (np.ndarray): watershed labels
        markers (np.ndarray): watershed markers

    Return:
        np.array: binary image of watershed labels separated by contours
    """

    black = np.zeros(img.shape, np.uint8)

    for label in np.unique(labels):
        if label == 0:
            continue
        # Create a mask
        mask = np.zeros(markers.shape, dtype="uint8")
        mask[labels == label] = 255
        # Find contours and determine contour area
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        c = max(cnts, key=cv2.contourArea)
        cv2.drawContours(black, [c], -1, (255, 255, 255), -1)
        cv2.drawContours(black, [c], -1, (0, 0, 0), 2)
    cnts = cv2.findContours(markers.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    cv2.drawContours(black, cnts, -1, (255, 0, 0), -1)
    return black


"""
Implementation of methods used to extract markers from bubble images.
"""

import sys

import numpy as np
import cv2

sys.path.append('..')


def _get_markers(img: np.array) -> np.array:
    """
    The threshold is set for a preprocessed image and is not meant to be changed.

    Args:
        img (np.array): processed image

    Returns:
        np.array: binary image of thresholded markers
    """
    _, all = cv2.threshold(img, 60, 255, cv2.THRESH_BINARY)
    return all


def _get_big_markers(markers: np.array) -> np.array:
    """
    Extract big bubble markers by morphological operations.

    Args:
        markers (np.array): binary mask of all markers

    Returns:
        np.array: big markers
    """
    kernel_b = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    big = cv2.erode(markers, kernel_b, iterations=3)
    big = cv2.dilate(big, kernel_b, iterations=2)
    return big


def _get_small_markers(markers: np.array) -> np.array:
    """
    Extract small bubbles as difference between all bubble markers and big markers.

    Args:
        markers (np.array): binary mask of all markers

    Returns:
        np.array: small bubles mask
    """
    big = cv2.dilate(_get_big_markers(markers),
                     kernel=np.ones((5, 5), np.uint8),
                     iterations=6)
    inv_big = 255 - big
    return np.where(inv_big == 255, markers, 0)


def get_markers(marker_type: str, img: np.array) -> np.array:
    """
    Extract bubble markers by threshold.

    Args:
        marker_type (str): 'all', 'big', 'small'
        img (np.array): grayscale image to threshold. should be preprocessed

    Returns:
        np.array: binary image of markers
    """

    markers = _get_markers(img)

    if marker_type == 'all':
        return markers
    elif marker_type == 'big':
        return _get_big_markers(markers)
    elif marker_type == 'small':
        return _get_small_markers(markers)
