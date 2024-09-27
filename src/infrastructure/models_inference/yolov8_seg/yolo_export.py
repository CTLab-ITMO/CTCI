"""
Model Export Script

This script is used for exporting a trained segmentation model to ONNX format and optionally quantizing the model.

The script takes a YAML configuration file as input, which specifies the paths to the trained model checkpoint,
input tensor shape, and others configuration parameters.

Usage:
    python <model>_export.py <config_path>

Args:
    config_path (str): Path to the YAML configuration file containing model export parameters.
"""


import sys

from src.models.yolov8_segmentation.yolov8 import load_yolov8_segment
from src.models.utils.config import read_yaml_config


if __name__ == "__main__":
    config_path = sys.argv[1]
    config_data = read_yaml_config(config_path)

    model_path = config_data['checkpoint_path']
    model = load_yolov8_segment(model_path)

    input_tensor_shape = config_data['input_tensor_shape']

    model.export(
        format='onnx',
        imgsz=input_tensor_shape[-1],
        half=config_data['export']['half'],
        simplify=config_data['export']['simplify']
    )
