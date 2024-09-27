import torch.nn as nn
from ultralytics import YOLO


def load_yolov8_segment(checkpoint_path: str):
    segment = YOLO(checkpoint_path)
    return segment


def init_yolov8_segment(model_type):
    model_types = {
        'nano': 'yolov8n-seg.pt',
        'small': 'yolov8s-seg.pt',
        'medium': 'yolov8m-seg.pt',
        'large': 'yolov8l-seg.pt',
        'extra': 'yolov8x-seg.pt'
    }

    if model_type.split('.')[-1] != 'pt':
        model_type = model_types[model_type]

    model = YOLO(model_type)
    return model


def train_yolov8_segment(model, config_data):
    metrics = model.train(
        data=config_data['data'],
        batch=config_data['batch_size'],
        save_json=config_data['save_json'],
        epochs=config_data['num_epochs'],
        pretrained=config_data['pretrained']
    )

    return metrics
