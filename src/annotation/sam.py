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
from src.utils.masks import masks_narrowing, unite_masks
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


def yolo_sam_segmentation(
        image, detector: YOLO, predictor: SamPredictor, wshed: Watershed,
        target_length: int = 1024, narrowing: float = 0.20, erode_iterations: float = 1,
        prompt_points: bool = True
):
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

    Returns:
        numpy.ndarray: Segmented mask combining YOLOv8 and SAM predictions.

    Notes:
        - The function detects objects using YOLOv8 and performs segmentation using SAM.
        - SAM segmentation is applied only if YOLOv8 detects objects.
        - The final mask is a combination of SAM and watershed segmentation.
        - The mask is further eroded using morphological operations.

    Example:
        mask = yolo_sam_segmentation(image, yolo_detector, sam_predictor)
    """
    boxes = yolov8_detect(image=image, detector=detector, return_objects=False)
    mask_watershed = wshed.apply_watershed(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

    if len(boxes) != 0:
        masks_list = sam_segmentation(image=image, predictor=predictor, boxes=boxes, prompt_points=prompt_points,
                                      target_length=target_length)
        masks_united = []
        for masks in masks_list:
            masks_united.append(unite_masks(masks))
        masks_narrowed = masks_narrowing(masks_united, narrowing=narrowing)
        mask_sam = unite_masks(masks_narrowed)
    else:
        mask_sam = np.zeros_like(mask_watershed)

    mask = unite_masks([mask_sam, mask_watershed])

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_final = cv2.erode(mask, kernel, iterations=erode_iterations)

    return mask_final


def segment_image_from_dir(
        image_name: str,
        masks_list,
        source_dir: str,
        output_dir: str,
        detector: YOLO, predictor: SamPredictor, wshed: Watershed,
        target_length: int = 1024,
        narrowing: float = 0.20,
        erode_iterations: int = 1,
        prompt_points=True
):
    """
    Perform segmentation on an image from a source directory and save the result to an output directory.

    Args:
        image_name (str): Name of the image file to be segmented.
        masks_list (list): List of already segmented image names to avoid duplicate segmentation.
        source_dir (str): Directory containing the source images.
        output_dir (str): Directory where the segmented masks will be saved.
        detector: YOLOv8 object detector.
        predictor (SamPredictor): Instance of the predictor class for the SAM model.
        target_length (int, optional): Target length for resizing the longest side of the image.
            Defaults to 1024 pixels.
        narrowing (float, optional): Narrowing factor for SAM masks. Defaults to 0.20.
        erode_iterations (int, optional): Number of iterations for eroding the final mask. Defaults to 1.

    Returns:
        None

    Notes:
        - If the image has already been segmented (present in masks_list), the function returns early.
        - Prints progress information during segmentation.
        - Saves the segmented mask to the output directory with the same name as the input image.
    """
    if image_name in masks_list:
        return

    print(f"Image {image_name}")
    image = cv2.imread(os.path.join(source_dir, image_name))
    mask = yolo_sam_segmentation(
        image, detector, predictor, wshed,
        target_length=target_length,
        narrowing=narrowing,
        erode_iterations=erode_iterations,
        prompt_points=prompt_points
    )

    cv2.imwrite(
        filename=os.path.join(output_dir, image_name),
        img=mask
    )

    print(f"Image {image_name} were segmented!")


def segment_images_from_folder(
        source_dir,
        output_dir,
        detector, predictor, wshed,
        target_length=1024,
        narrowing=0.20,
        erode_iterations=1,
        processes_num=0,
        prompt_points=True,
):
    """
    Perform segmentation on images in a source directory and save the results to an output directory.

    Args:
        source_dir (str): Directory containing the source images.
        output_dir (str): Directory where the segmented masks will be saved.
        detector: YOLOv8 object detector.
        predictor (SamPredictor): Instance of the predictor class for the SAM model.
        target_length (int, optional): Target length for resizing the longest side of the image.
            Defaults to 1024 pixels.
        narrowing (float, optional): Narrowing factor for SAM masks. Defaults to 0.20.
        erode_iterations (int, optional): Number of iterations for eroding the final mask. Defaults to 1.
        processes_num (int, optional): Number of parallel processes for segmentation.
            Defaults to 0 (sequential processing).

    Returns:
        None

    Notes:
        - Prints progress information during segmentation.
        - Saves the segmented masks to the output directory.
        - If processes_num is set to 0, performs sequential segmentation (allows using debug mode).
        - If processes_num is greater than 0, performs parallel segmentation using concurrent.futures.
    """
    if not os.path.isdir(source_dir):
        print("No such directory")
        return "No such directory"

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    images_list = os.listdir(source_dir)
    masks_list = os.listdir(output_dir)

    for image_name in tqdm(images_list):
        with torch.no_grad():
            segment_image_from_dir(
                image_name,
                masks_list,
                source_dir,
                output_dir,
                detector, predictor, wshed,
                target_length=target_length,
                narrowing=narrowing,
                erode_iterations=erode_iterations,
                prompt_points=prompt_points
            )

    print(f"Images from the directory {source_dir} were segmented!")
