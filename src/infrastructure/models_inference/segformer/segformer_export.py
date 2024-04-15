import sys

import transformers

from src.models.segformer.model import SegFormer
from src.models.inference import export_model_onnx, quantize_onnx
from src.models.utils.config import read_yaml_config


if __name__ == "__main__":
    config_path = sys.argv[1]
    config_handler = read_yaml_config(config_path)

    # TODO: load checkpoints
    net = transformers.SegformerForSemanticSegmentation.from_pretrained(
        f"nvidia/segformer-b2-finetuned-ade-512-512",
        num_labels=1,
        image_size=256,
        ignore_mismatched_sizes=True
    )
    model = SegFormer(net)

    input_tensor_shape = config_handler.read('input_tensor_shape')

    export_model_onnx(
        model,
        config_handler=config_handler
    )

    if config_handler.read('acceleration', 'quantization'):
        quantize_onnx(config_handler)

