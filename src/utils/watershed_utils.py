"""
Utilities for watershed.
"""
import numpy as np
import cv2


def find_contours(img: np.ndarray, labels: np.ndarray, markers: np.ndarray) -> np.ndarray:
    """
    Draw a binary mask of contours between watershed labels.

    Args:
        img (np.ndarray): Grayscale image.
        labels (np.ndarray): Watershed labels.
        markers (np.ndarray): Watershed markers.

    Returns:
        np.ndarray: Binary image of watershed labels separated by contours.
    """

    # Initialize an empty binary mask of the same size as the input image
    contour_mask = np.zeros(img.shape, dtype=np.uint8)

    # Iterate over unique labels in the watershed result
    unique_labels = np.unique(labels)

    for label in unique_labels:
        if label == 0:
            continue  # Skip the background label

        # Create a mask for the current label
        label_mask = np.zeros(markers.shape, dtype="uint8")
        label_mask[labels == label] = 255

        # Find contours in the mask
        contours = cv2.findContours(label_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        # Draw the largest contour by area
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(contour_mask, [largest_contour], -1, (255, 255, 255),
                             thickness=cv2.FILLED)  # Fill the contour
            cv2.drawContours(contour_mask, [largest_contour], -1, (0, 0, 0), thickness=2)  # Draw contour outline

    # Draw external contours from the marker image
    marker_contours, _ = cv2.findContours(markers.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(contour_mask, marker_contours, -1, (255, 0, 0), thickness=cv2.FILLED)

    return contour_mask


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
