import sys

import numpy

import onnx

from src.models.inference import init_onnx_session
from src.models.utils.config import read_yaml_config
from src.infrastructure.models_inference.video_inference import video_onnx_inference


if __name__ == "__main__":
    config_path = sys.argv[1]
    config_handler = read_yaml_config(config_path)

    session = init_onnx_session(config_handler)
    video_onnx_inference("..\\data\\test_data\\bubbles\\video_0.mp4", session)

