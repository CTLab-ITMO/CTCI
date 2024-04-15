import sys

import numpy

import onnx

from src.models.inference import init_onnx_session
from src.models.utils.config import read_yaml_config


if __name__ == "__main__":
    config_path = sys.argv[1]
    config_handler = read_yaml_config(config_path)

    session = init_onnx_session(config_handler)

    inputs = {'input': numpy.ones(shape=(2, 3, 256, 256), dtype=numpy.float32)}
    outputs = session.run(None, inputs)
    print(type(outputs[0]))
