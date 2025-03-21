"""
Watershed segmentation for small bubbles extraction is implemented.
"""
import numpy as np
import cv2
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from src.annotation.preprocess import preprocess
from src.utils.watershed_utils import get_markers, find_contours

from omegaconf import DictConfig
from hydra import initialize, compose


class Watershed:
    def __init__(self, cfg: DictConfig):

        self.cfg = cfg
        self.use_preprocess = "preprocess" in cfg
        self.marker_type = self.cfg.marker_type
        self.min_distance = cfg.min_distance
        self.watershed_label_type = cfg.watershed_label_type

    def apply_watershed(self, img: np.array) -> np.array:
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
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if self.use_preprocess:
            gray_img = preprocess(gray_img, self.cfg.preprocess)

        smkr = get_markers(self.marker_type, gray_img, self.cfg.thresh)
        bsmkr = cv2.dilate(smkr,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 5)),
                           iterations=5)
        data = [smkr, smkr]

        if self.watershed_label_type == "marker_area":
            data = [smkr, bsmkr]
        elif self.watershed_label_type == "image":
            data = [bsmkr, gray_img]

        labels, markers = _apply_watershed(data=data, min_distance=self.min_distance)
        black = find_contours(gray_img, labels, markers)

        return black


def _apply_watershed(data: list, min_distance: int) -> tuple:
    """
    Extract labels for bubbles using watershed algorithm.

    Args:
        data: markers & unknown area

    Returns:
        tuple of np.array: labels, markers
    """
    markers, area = data
    distance_map = ndimage.distance_transform_edt(markers)
    local_max = peak_local_max(distance_map, min_distance=min_distance, labels=markers)
    peak_mask = np.zeros(distance_map.shape, dtype=bool)
    peak_mask[tuple(local_max.T)] = True
    peak_markers = ndimage.label(peak_mask)[0]
    labels = watershed(-distance_map, peak_markers, mask=area)
    return labels, markers


def init_watershed(config_path='../configs/watershed', config_name='watershed', version_base=None):
    with initialize(version_base=version_base, config_path=config_path):
        wshed = Watershed(cfg=compose(config_name))
    return wshed
