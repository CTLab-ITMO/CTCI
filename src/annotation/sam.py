"""
sam.py

This module provides functions for image segmentation using YOLOv8 object detection
and the Segment Anything Model (SAM). The functions perform segmentation on individual
images and process entire folders of images in parallel or sequentially.

Functions:
    - load_sam_predictor: Loads a SAM model from a checkpoint file and creates a predictor instance.
    - sam_segmentation: Performs segmentation using a SAM predictor on a given image with bounding boxes.
    - yolo_sam_segmentation: Performs image segmentation using YOLOv8 and SAM.
    - segment_image_from_dir: Segments an image from a source directory and saves the result to an output directory.
    - segment_images_from_folder: Segments images in a source directory and saves the results to an output directory.

Notes:
    - The SAM model is loaded and utilized through the provided predictor.
    - YOLOv8 is used for object detection before SAM segmentation.
    - SAM segmentation is applied only if YOLOv8 detects objects in the image.
    - Segmentation results are saved as masks in the output directory.
    - Parallel processing is supported using concurrent.futures (only on cpu).
"""

import os.path
import concurrent.futures

import cv2
import numpy as np
import torch

from tqdm import tqdm

from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide

from ultralytics import YOLO

from src.annotation.watershed import Watershed
from src.utils.masks import masks_narrowing, unite_masks, suppress_watershed_with_yolosam
from src.annotation.yolo import yolov8_detect


def load_sam_predictor(checkpoint_path: str, model_type: str, device: str = "cpu") -> SamPredictor:
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
    sam = sam.to(device)
    predictor = SamPredictor(sam)
    print(f"Predictor's device is {predictor.device}")

    return predictor


def sam_segmentation(
        image, predictor: SamPredictor,
        boxes: list, prompt_points: bool = True, target_length: int = 1024
):
    """
    Performs segmentation using a Segment Anything Model (SAM) predictor on the given image with the given prompts.

    Args:
        image (numpy.ndarray): The input image for segmentation.
        predictor (SamPredictor): An instance of the predictor class for the SAM model.
        boxes (list): A list of bounding boxes represented as [x_min, y_min, x_max, y_max].
        prompt_points (bool, optional): If True, prompts a point for each bounding box for segmentation.
            If False, performs segmentation without prompting a point. Default is True.
        target_length (int, optional): The target length for resizing the longest side of the image.
            Defaults to 1024 pixels.

    Returns:
        numpy.ndarray: A list of segmentation masks corresponding to each bounding box.

    Notes:
        The function uses the SAM model through the provided predictor to perform segmentation.
        If prompt_points is True, a point is prompted for each bounding box to aid segmentation.
    """
    device = predictor.device

    original_image_size = (image.shape[0], image.shape[1])
    transform = ResizeLongestSide(target_length=target_length)

    transformed_image = transform.apply_image(image)
    transformed_image_torch = torch.as_tensor(transformed_image, device=device)
    transformed_image_torch = transformed_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

    predictor.set_torch_image(
        transformed_image=transformed_image_torch,
        original_image_size=original_image_size
    )

    boxes = np.array(boxes)
    boxes = transform.apply_boxes(boxes, (predictor.original_size[0], predictor.original_size[1]))
    boxes_tensor = torch.Tensor(boxes).float().to(device)

    if prompt_points:
        points = [np.array([[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]]) for box in boxes]
        points = np.array(points)
        points = transform.apply_coords(points, (predictor.original_size[0], predictor.original_size[1]))
        points_tensor = torch.Tensor(points).float().to(device)
        labels_tensor = torch.ones((points.shape[0], points.shape[1])).float().to(device)
        masks_list, _, _ = predictor.predict_torch(
            boxes=boxes_tensor,
            point_coords=points_tensor,
            point_labels=labels_tensor,
            multimask_output=False
        )

    else:
        masks_list, _, _ = predictor.predict_torch(
            boxes=boxes_tensor,
            multimask_output=False,
            point_coords=None,
            point_labels=None
        )

    masks_list = masks_list.float().to("cpu").numpy()
    return masks_list


def watershed_segmentation(image: np.ndarray, wshed: Watershed) -> np.ndarray:
    """
    Perform image segmentation using the Watershed algorithm.

    Args:
        image (numpy.ndarray): Input grayscale image for segmentation.
        wshed (Watershed): Instance of the Watershed class.

    Returns:
        numpy.ndarray: Segmented mask using the Watershed algorithm.
    """
    return wshed.apply_watershed(image)


def yolo_sam_segmentation(
        image: np.ndarray, detector: YOLO, predictor: SamPredictor,
        target_length: int = 1024, narrowing: float = 0.20,
        erode_iterations: int = 1, prompt_points: bool = True
) -> np.ndarray:
    """
    Perform image segmentation using YOLOv8 and Segment Anything Model (SAM).

    Args:
        image (numpy.ndarray): Input image for segmentation.
        detector (YOLO): YOLOv8 object detector.
        predictor (SamPredictor): Instance of the predictor class for the SAM model.
        target_length (int, optional): Target length for resizing the longest side of the image.
            Defaults to 1024 pixels.
        narrowing (float, optional): Narrowing factor for SAM masks. Defaults to 0.20.
        erode_iterations (int, optional): Number of iterations for eroding the final mask. Defaults to 1.
        prompt_points (bool, optional): Whether to use prompt points for SAM. Defaults to True.

    Returns:
        numpy.ndarray: Segmented mask combining YOLOv8 and SAM predictions.

    Notes:
        - Detects objects using YOLOv8 and performs segmentation using SAM.
        - SAM segmentation is applied only if YOLOv8 detects objects.
        - The mask is further eroded using morphological operations.

    Example:
        mask = yolo_sam_segmentation(image, yolo_detector, sam_predictor)
    """
    boxes = yolov8_detect(image=image, detector=detector, return_objects=False)

    if len(boxes) != 0:
        masks_list = sam_segmentation(image=image, predictor=predictor, boxes=boxes, prompt_points=prompt_points,
                                      target_length=target_length)
        masks_united = [unite_masks(masks) for masks in masks_list]
        masks_narrowed = masks_narrowing(masks_united, narrowing=narrowing)
        mask_sam = unite_masks(masks_narrowed)
    else:
        mask_sam = np.zeros(image.shape[:2], dtype=np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_final = cv2.erode(mask_sam, kernel, iterations=erode_iterations)

    return mask_final


def combined_segmentation(
    image, detector=None, predictor=None, wshed=None,
    target_length=None, narrowing=None, erode_iterations=None, prompt_points=None,
    combination_type="unite"
) -> np.ndarray:
    """
    Combine YOLO + SAM and Watershed segmentation results based on availability.

    Args:
        image (numpy.ndarray): Input image for segmentation.
        detector (YOLO, optional): YOLOv8 object detector. Defaults to None.
        predictor (SamPredictor, optional): SAM predictor instance. Defaults to None.
        wshed (Watershed, optional): Watershed instance. Defaults to None.
        target_length (int, optional): Target length for resizing in SAM. Defaults to None.
        narrowing (float, optional): Narrowing factor for SAM masks. Defaults to None.
        erode_iterations (int, optional): Number of erosion iterations for masks. Defaults to None.
        prompt_points (bool, optional): Whether to use prompt points in SAM. Defaults to None.

    Returns:
        numpy.ndarray: Combined segmentation mask.

    Raises:
        ValueError: If no valid segmentation method is provided.
    """
    mask_sam = None
    mask_watershed = None

    if detector and predictor:
        mask_sam = yolo_sam_segmentation(
            image=image, detector=detector, predictor=predictor,
            target_length=target_length, narrowing=narrowing,
            erode_iterations=erode_iterations, prompt_points=prompt_points
        )

    if wshed:
        mask_watershed = watershed_segmentation(image=image, wshed=wshed)

    if mask_sam is not None and mask_watershed is not None:
        if combination_type == "unite":
            return unite_masks([mask_sam, mask_watershed])
        elif combination_type == "sam_first":
            mask_watershed = suppress_watershed_with_yolosam(mask_sam, mask_watershed)
            return unite_masks([mask_sam, mask_watershed])
    elif mask_sam is not None:
        return mask_sam
    elif mask_watershed is not None:
        return mask_watershed
    else:
        raise ValueError("No valid segmentation method provided.")


def segment_images_from_folder(
    source_dir, output_dir, detector=None, predictor=None, wshed=None,
    target_length=None, narrowing=None, erode_iterations=None, prompt_points=None, combination_type="unite"
):
    """
    Segments images in a folder using combined YOLO + SAM and Watershed segmentation.

    Args:
        source_dir (str): Path to the input images directory.
        output_dir (str): Path to save the segmented masks.
        detector (YOLO, optional): YOLO object detector instance.
        predictor (SamPredictor, optional): SAM predictor instance.
        wshed (Watershed, optional): Watershed instance.
        target_length (int, optional): Target length for resizing in SAM. Defaults to None.
        narrowing (float, optional): Narrowing factor for SAM masks. Defaults to None.
        erode_iterations (int, optional): Number of erosion iterations for masks. Defaults to None.
        prompt_points (bool, optional): Whether to use prompt points in SAM. Defaults to None.

    Returns:
        None: Saves segmented masks to the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    if detector is None and predictor is None and wshed is None:
        raise ValueError("At least one segmentation method (YOLO + SAM or Watershed) must be provided.")

    for image_name in os.listdir(source_dir):
        image_path = os.path.join(source_dir, image_name)
        image = cv2.imread(image_path)

        # Perform combined segmentation
        mask = combined_segmentation(
            image, detector=detector, predictor=predictor, wshed=wshed,
            target_length=target_length, narrowing=narrowing,
            erode_iterations=erode_iterations, prompt_points=prompt_points, combination_type=combination_type,
        )

        # Save the generated mask
        mask_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}_mask.png")
        cv2.imwrite(mask_path, mask)
