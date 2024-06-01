"""
yolov8.py

This module provides functions for working with YOLOv8 object detection.
"""

import matplotlib.pyplot as plt
import cv2

from ultralytics import YOLO


def load_yolov8_detector(checkpoint_path: str):
    """
     Loads a YOLO object detector from the specified path.

    Args:
        checkpoint_path (str): The path to the YOLO weights.

    Returns:
        YOLO: A YOLO object detector initialized with the provided  weights.

    """
    detector = YOLO(checkpoint_path)
    return detector


def yolov8_detect(image, detector, return_objects=False):
    """
    Performs object detection using a YOLOv8 detector on the given image.

    Args:
        image: The input image for object detection.
        detector (YOLO): The YOLOv8 object detector.
        return_objects (bool, optional): If True, returns the raw detected objects.
            If False (default), returns a list of bounding boxes.

    Returns:
        list of YOLOObjects: If return_objects is True, returns the raw detected objects.
        If return_objects is False, returns a list of bounding boxes represented as lists
        [x_min, y_min, x_max, y_max], where (x_min, y_min) are the coordinates of the top-left
        corner and (x_max, y_max) are the coordinates of the bottom-right corner.

    """
    objects = detector(image)

    if return_objects:
        return objects

    boxes = objects[0].boxes.xyxy.tolist()

    return boxes


def draw_bounding_boxes(image, boxes, cmap=None, edge_color='red', face_color=(0, 0, 0, 0), lw=2):
    """
    Draws bounding boxes on the given image.

    Args:
        image (numpy.ndarray): The input image.
        boxes (list): A list of bounding boxes represented as [x_min, y_min, x_max, y_max].
        cmap (str or None, optional): Colormap for displaying the image.
        edge_color (str or tuple, optional): Color of the bounding box edges.
        face_color (str or tuple, optional): Color of the bounding box face (fill color).
        lw (int, optional): Line width of the bounding box edges.

    Returns:
        None
    """
    for box in boxes:
        x1, y1, x2, y2 = box

        w, h = x2 - x1, y2 - y1
        plt.Rectangle((x1, y1), w, h, edgecolor=edge_color, facecolor=face_color, lw=lw)
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

    plt.imshow(image, cmap=cmap)
    plt.axis('off')
    plt.show()
