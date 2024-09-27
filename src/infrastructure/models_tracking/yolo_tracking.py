"""
YOLOv8 Segmentation Experiment Script

This script is used for running a segmentation experiment using the YOLOv8 architecture.
It loads or initializes a YOLOv8-based segmentation model, trains the model based on the provided configuration file,
    and prints out the metrics obtained during training.

Usage:
    python yolo_tracking.py <config_path>
"""

import sys

from src.models.yolov8_segmentation.yolov8 import load_yolov8_segment, init_yolov8_segment, train_yolov8_segment
from src.models.utils.config import read_yaml_config


if __name__ == "__main__":
    config_path = sys.argv[1]
    config_handler = read_yaml_config(config_path)

    checkpoint_path = config_handler.read('checkpoint_path')
    model_type = config_handler.read('model_type')
    if checkpoint_path == 'None':
        model = init_yolov8_segment(model_type)
    else:
        model = load_yolov8_segment(checkpoint_path)

    metrics = train_yolov8_segment(model, config_data=config_handler)
    print(metrics)
