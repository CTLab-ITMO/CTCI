import sys
sys.path.append('../..')

from src.models.watershed.preprocessing import *


def _get_markers(img: np.array):
    i = preprocess(img)
    _, all = cv2.threshold(i, 60, 255, cv2.THRESH_BINARY)
    return all

def _get_big_markers(markers):
    kernel_b = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    big = cv2.erode(markers, kernel_b, iterations=3)
    big = cv2.dilate(big, kernel_b, iterations=2)
    return big 

def _get_small_markers(markers):
    big = cv2.dilate(_get_big_markers(markers),
                     kernel = np.ones((5,5), np.uint8),
                     iterations=6)
    inv_big = 255 - big
    return np.where(inv_big == 255, markers, 0)

def get_markers(marker_type: str, img: np.array):
    markers = _get_markers(img)

    if marker_type=='all':
        return markers
    elif marker_type=='big':
        return _get_big_markers(markers)
    elif marker_type=='small':
        return _get_small_markers(markers)