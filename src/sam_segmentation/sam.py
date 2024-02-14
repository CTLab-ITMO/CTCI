"""
sam.py

This module provides functions for working with the Segment Anything Model (SAM) and performing segmentation.

Dependencies:
- numpy as np
- segment_anything.sam_model_registry
- segment_anything.SamPredictor

Usage:
- load_sam_predictor(checkpoint_path, model_type): Loads a SAM model from a checkpoint and creates a predictor instance.
- sam_segmentation(image, predictor, boxes, prompt_point=True): Performs segmentation on an image using a SAM predictor
and bounding boxes.

Notes:
- The module assumes the existence of sam_model_registry and SamPredictor classes from the 'segment_anything' package.

"""

import numpy as np

from segment_anything import sam_model_registry, SamPredictor


def load_sam_predictor(checkpoint_path: str, model_type: str, device: str = "cuda") -> SamPredictor:
    """

    Loads a Segment Anything Model (SAM) from a checkpoint file and creates a predictor instance.

    Args:
        checkpoint_path (str): The path to the checkpoint file for the SAM model.
        model_type (str): The type or variant of the SAM model to be loaded.
        device (str, optional): The device to which the SAM model should be loaded.
            Defaults to "cuda" (GPU) if available, otherwise falls back to "cpu".

    Returns:
        SamPredictor: An instance of the predictor class for the SAM model.
    """
    sam = sam_model_registry[model_type](checkpoint_path)
    predictor = SamPredictor(sam)

    return predictor


def sam_segmentation(image, predictor, boxes, prompt_point=True):
    """
    Performs segmentation on the given image using a Segment Anything Model (SAM) predictor
    and bounding boxes.

    Args:
        image: The input image for segmentation.
        predictor (SamPredictor): An instance of the predictor class for the SAM model.
        boxes (list): A list of bounding boxes represented as [x_min, y_min, x_max, y_max].
        prompt_point (bool, optional): If True, prompts a point for each bounding box for segmentation.
            If False, performs segmentation without prompting a point. Default is True.
    """
    predictor.set_image(image)
    masks_list = []
    for box in boxes:
        box = np.array(box)
        if prompt_point:
            point = np.array([[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]])
            labels = np.array([1])
            masks, _, _ = predictor.predict(
                box=box,
                point_coords=point,
                point_labels=labels,
                multimask_output=False
            )
        else:
            masks, _, _ = predictor.predict(
                box=box,
                multimask_output=False
            )
        masks_list.append(masks)

    return np.array(masks_list)

