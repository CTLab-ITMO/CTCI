import sys

import torch
import numpy

from src.models.inference import init_onnx_session
from src.models.utils.config import read_yaml_config

# Example code snippet
# TODO: make method for runtime inference
if __name__ == "__main__":
    config_path = sys.argv[1]
    config_data = read_yaml_config(config_path)

    session = init_onnx_session(config_data)

    inputs = {'input': numpy.ones(shape=(2, 3, 256, 256), dtype=numpy.float32)}
    outputs = session.run(None, inputs)
