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
