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
import os.path
import concurrent.futures

import cv2
import numpy as np
import torch
from tqdm import tqdm

from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide

from src.processing.watershed import perform_watershed
from src.sam_segmentation.utils import masks_narrowing, unite_masks
from src.sam_segmentation.yolo import yolov8_detect


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


def sam_segmentation(image, predictor, boxes, prompt_points=True, target_length=800):
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
            multimask_output=False
        )

    masks_list = masks_list.float().to("cpu").numpy()
    return masks_list


def yolo_sam_segmentation(image, detector, predictor, target_length=800, narrowing=0.20, erode_iterations=1):
    boxes = yolov8_detect(image=image, detector=detector, return_objects=False)
    mask_watershed = perform_watershed(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

    if len(boxes) != 0:
        masks_list = sam_segmentation(image=image, predictor=predictor, boxes=boxes, prompt_points=True,
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
        image_name,
        masks_list,
        source_dir,
        output_dir,
        detector, predictor,
        target_length=800,
        narrowing=0.20,
        erode_iterations=1
):
    if image_name in masks_list:
        return

    print(f"Image {image_name}")
    image = cv2.imread(os.path.join(source_dir, image_name))
    mask = yolo_sam_segmentation(
        image, detector, predictor,
        target_length=target_length,
        narrowing=narrowing,
        erode_iterations=erode_iterations
    )

    cv2.imwrite(
        filename=os.path.join(output_dir, image_name),
        img=mask
    )

    print(f"Image {image_name} were segmented!")


def segment_images_from_folder(
        source_dir,
        output_dir,
        detector, predictor,
        target_length=800,
        narrowing=0.20,
        erode_iterations=1,
        processes_num=0
):
    if not os.path.isdir(source_dir):
        print("No such directory")
        return "No such directory"

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    images_list = os.listdir(source_dir)
    masks_list = os.listdir(output_dir)

    if processes_num == 0:
        for image_name in tqdm(images_list):
            with torch.no_grad():
                segment_image_from_dir(
                    image_name,
                    masks_list,
                    source_dir,
                    output_dir,
                    detector, predictor,
                    target_length=target_length,
                    narrowing=narrowing,
                    erode_iterations=erode_iterations
                )
    else:
        with torch.no_grad():
            executor = concurrent.futures.ProcessPoolExecutor(processes_num)
            futures = [
                executor.submit(
                    segment_image_from_dir, image_name, masks_list,
                    source_dir, output_dir,
                    detector, predictor,
                    target_length, narrowing, erode_iterations
                ) for image_name in images_list
            ]
            concurrent.futures.wait(futures)

    print(f"Images from the directory {source_dir} were segmented!")
