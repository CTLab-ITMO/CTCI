import sys

from src.models.yolov8_segmentation.model import load_yolov8_segment
from src.models.inference import export_model_onnx
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
