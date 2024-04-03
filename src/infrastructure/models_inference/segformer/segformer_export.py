import sys

import transformers

from src.models.segformer.model import SegFormer
from src.models.inference import export_model_onnx, quantize_onnx
from src.models.utils.config import read_yaml_config


if __name__ == "__main__":
    config_path = sys.argv[1]
    config_data = read_yaml_config(config_path)

    # model_path = config_data['checkpoint_path']
    # TODO: load checkpoints
    net = transformers.SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b1-finetuned-ade-512-512")
    image_processor = transformers.SegformerImageProcessor()
    model = SegFormer(net=net, image_processor=image_processor)

    input_tensor_shape = config_data['input_tensor_shape']

    export_model_onnx(
        model,
        config_data=config_data
    )

    if config_data['acceleration']['quantization']:
        quantize_onnx(config_data)

