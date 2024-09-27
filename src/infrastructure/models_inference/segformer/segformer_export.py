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

import torch
import transformers

from src.models.segformer.segformer import SegFormer
from src.models.inference import export_model_onnx, quantize_onnx
from src.models.utils.config import read_yaml_config


if __name__ == "__main__":
    config_path = sys.argv[1]
    config_handler = read_yaml_config(config_path)

    model_checkpoint_path = config_handler.read("checkpoint_path")
    net = transformers.SegformerForSemanticSegmentation.from_pretrained(
        f"nvidia/segformer-b2-finetuned-ade-512-512",
        num_labels=1,
        image_size=256,
        ignore_mismatched_sizes=True
    )

    model = SegFormer(net)
    model.load_state_dict(torch.load(model_checkpoint_path))
    input_tensor_shape = config_handler.read('input_tensor_shape')

    export_model_onnx(
        model,
        config_handler=config_handler
    )

    if config_handler.read('acceleration', 'quantization'):
        quantize_onnx(config_handler)

