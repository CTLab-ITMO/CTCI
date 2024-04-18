import sys

import torch
import transformers

from src.models.swin.model import Swin
from src.models.inference import export_model_onnx, quantize_onnx
from src.models.utils.config import read_yaml_config


if __name__ == "__main__":
    config_path = sys.argv[1]
    config_handler = read_yaml_config(config_path)

    model_checkpoint_path = config_handler.read("checkpoint_path")
    net = transformers.Swinv2Model.from_pretrained(model_checkpoint_path)

    model = Swin(net)
    model = model.load_state_dict(torch.load(model_checkpoint_path))

    input_tensor_shape = config_handler.read('input_tensor_shape')

    export_model_onnx(
        model,
        config_handler=config_handler
    )

    if config_handler.read('acceleration', 'quantization'):
        quantize_onnx(config_handler)
