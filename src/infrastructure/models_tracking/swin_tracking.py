import sys
import os.path as osp

import albumentations as albu
from transformers import Swinv2Model

from src.features.segmentation.dataset import SegmentationDataset
from src.models.swin.model import Swin
from src.infrastructure.tracking import tracking_experiment
from src.models.utils.config import read_yaml_config

sys.path.append('../../models')


if __name__ == "__main__":
    config_path = sys.argv[1]
    config_handler = read_yaml_config(config_path)

    model_name = config_handler.read('model', 'model_name')
    model_type = config_handler.read('model', 'model_type')

    model_str = rf"microsoft/{model_name}-{model_type}"
    net = Swinv2Model.from_pretrained(model_str)

    model = Swin(net=net)

    transform = albu.Compose([
        albu.Resize(config_handler.read('dataset', 'image_size', 'height'), config_handler.read('dataset', 'image_size', 'width')),
        albu.CLAHE(always_apply=True),
        albu.Normalize(always_apply=True),
        albu.RandomCrop(config_handler.read('dataset', 'image_size', 'height'),
                        config_handler.read('dataset', 'image_size', 'width'), p=0.4),
        albu.MaskDropout(max_objects=10),
        albu.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=30, p=0.5)
    ])

    train_dataset = SegmentationDataset(
        images_dir=osp.join(config_handler.read('dataset', 'training_dataset_dirs')[0], "images"),
        masks_dir=osp.join(config_handler.read('dataset', 'training_dataset_dirs')[0], "masks"),
        augmentation_transform=transform
    )
    val_dataset = SegmentationDataset(
        images_dir=osp.join(config_handler.read('dataset', 'validation_dataset_dirs')[0], "images"),
        masks_dir=osp.join(config_handler.read('dataset', 'validation_dataset_dirs')[0], "masks"),
        augmentation_transform=transform
    )

    experiment_name = config_handler.read('mlflow', 'experiment_name')
    if experiment_name == "None":
        experiment_name = model_name

    tracking_experiment(
        model,
        train_dataset, val_dataset,
        config_handler,
        experiment_name=experiment_name
    )
