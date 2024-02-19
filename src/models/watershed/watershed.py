"""
Watershed segmentation for small bubbles extraction is implemented.
"""

import numpy as np
import cv2
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from features.preprocessing import preprocess
from models.watershed.markers import get_markers
from models.watershed.utils import find_contours


def _perform_watershed(data: list):
    """
    Extract labels for bubbles using watershed algorithm.
    
    Args:
        data: markers & unknown area

    Returns:
        tuple of np.array: labels, markers
    """
    markers, area = data
    distance_map = ndimage.distance_transform_edt(markers)
    local_max = peak_local_max(distance_map, min_distance=5, labels=markers)
    peak_mask = np.zeros(distance_map.shape, dtype=bool)
    peak_mask[tuple(local_max.T)] = True
    peak_markers = ndimage.label(peak_mask)[0]
    labels = watershed(-distance_map, peak_markers, mask=area)
    return labels, markers


def perform_watershed(img):
    """
    Apply watershed to the bubble image to extract small bubbles masks separated by
    contours. Image is converted to grayscale and processed. After that,
    small bubble markers are extracted and dilated to create an unknown area. 
    The watershed is performed over that.

    Args:
        img (np.array): image.

    Returns:
        np.array: image of bubble masks separated by contours
    """
    i = preprocess(img)
    smkr = get_markers('small', i)
    bsmkr = cv2.dilate(smkr,
                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,5)),
                   iterations=5)

    labels, markers = _perform_watershed(data=[smkr, bsmkr])
    black = find_contours(img, labels, markers)

    return black
