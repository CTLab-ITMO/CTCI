import numpy as np
import cv2
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import matplotlib.pyplot as plt
from src.processing.preprocessing import preprocess
from src.processing.contours import get_contours
from src.processing.markers import get_markers
from src.processing.utils import find_contours


def _small_watershed(data: list):
    markers, weak_contours = data
    distance_map = ndimage.distance_transform_edt(markers)
    local_max = peak_local_max(distance_map, min_distance=5, labels=markers)
    peak_mask = np.zeros(distance_map.shape, dtype=bool)
    peak_mask[tuple(local_max.T)] = True
    peak_markers = ndimage.label(peak_mask)[0]
    labels = watershed(-distance_map, peak_markers, mask=weak_contours)
    return labels, markers


def _big_watershed(data: list):
    markers, thresh = data
    distance_map = ndimage.distance_transform_edt(thresh)
    peak_markers = ndimage.label(markers, structure=np.ones((3,3)))[0]
    labels = watershed(-distance_map, peak_markers)
    return labels, markers


def _perform_watershed(data: list, bubble_size: str):
    """
    extract contours for bubbles
    
    args:
        bubble_size: 'big' or 'small'
        data: if 'big', thresholded big bubble markers & markers pic
                if 'small' markers and contours
    """
    if bubble_size=='small':
        return _small_watershed(data)
    elif bubble_size=='big':
        return _big_watershed(data)


def perform_watershed(img):
    """
    apply watershed to the bubble image to extract small bubbles
    args:
        img: grayscale np.array

    return:
        gray: grayscale image with contours
        black: binary image of contours
    """

    smkr = get_markers('small', img)
    bsmkr = cv2.dilate(smkr,
                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,5)),
                   iterations=5)

    labels, markers = _perform_watershed(data=[smkr, bsmkr], bubble_size='small')
    
    gray, black = find_contours(img, labels, markers)
    return gray, black
