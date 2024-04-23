import sys

from src.models.yolov8_segmentation.model import load_yolov8_segment, init_yolov8_segment, train_yolov8_segment
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
