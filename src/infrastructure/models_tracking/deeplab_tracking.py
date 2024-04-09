import sys
import os

sys.path.append(f"../src/")

import os.path as osp
import albumentations as albu

import segmentation_models_pytorch as smp

from src.features.segmentation.dataset import SegmentationDataset
from src.models.deeplab.model import DeepLab
from src.infrastructure.tracking import tracking_experiment
from src.models.utils.config import read_yaml_config


if __name__ == "__main__":
    config_path = sys.argv[1]
    config_data = read_yaml_config(config_path)

    model_name = config_data['model']['model_name']
    net = smp.DeepLabV3Plus(encoder_name=model_name)
    model = DeepLab(net=net)

    tr = albu.Compose([
        albu.Resize(config_data['dataset']['image_size']['height'], config_data['dataset']['image_size']['width']),
        albu.CLAHE(always_apply=True),
        albu.Normalize(always_apply=True),
        albu.RandomCrop(config_data['dataset']['image_size']['height'],
                        config_data['dataset']['image_size']['width'], p=0.4),
        albu.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=30, p=0.5)
    ])

    # TODO: create a func to init datasets from config_data
    train_dataset = SegmentationDataset(
        images_dir=osp.join(config_data['dataset']['training_dataset_dirs'][0], "images"),
        masks_dir=osp.join(config_data['dataset']['training_dataset_dirs'][0], "masks"),
        augmentation_transform=tr
    )

    val_dataset = SegmentationDataset(
        images_dir=osp.join(config_data['dataset']['validation_dataset_dirs'][0], "images"),
        masks_dir=osp.join(config_data['dataset']['validation_dataset_dirs'][0], "masks"),
        augmentation_transform=tr
    )

    experiment_name = config_data['mlflow']['experiment_name']
    if experiment_name == "None":
        experiment_name = model_name

    tracking_experiment(
        model,
        train_dataset, val_dataset,
        config_data,
        experiment_name=experiment_name
    )

