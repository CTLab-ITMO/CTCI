"""
Tracking Script

This script runs a segmentation experiment based on the provided configuration file.
It initializes a segmentation model, prepares the training and validation datasets,
    and tracks the experiment using MLflow.

Usage:
    python <model>_tracking.py <config_path>

Args:
    config_path (str): Path to the YAML configuration file containing experiment parameters.

"""

import sys

sys.path.append('../../models')

from src.features.segmentation.dataset import get_train_dataset_by_config, get_val_dataset_by_config
from src.models.swin.model import build_swin
from src.infrastructure.tracking import tracking_experiment
from src.features.segmentation.augmentation import get_augmentations_from_config, get_preprocess_from_config
from src.models.utils.config import read_yaml_config


if __name__ == "__main__":
    config_path = sys.argv[1]
    config_handler = read_yaml_config(config_path)

    model_name = config_handler.read('model', 'model_name')
    model = build_swin(config_handler)

    tr = get_preprocess_from_config(config_handler)
    aug = get_augmentations_from_config(config_handler)

    train_dataset = get_train_dataset_by_config(
        config_handler,
        transform=tr,
        augmentation_transform=aug
    )
    val_dataset = get_val_dataset_by_config(
        config_handler,
        transform=tr,
        augmentation_transform=aug
    )
    adele_dataset = get_train_dataset_by_config(
        config_handler,
        transform=tr,
        augmentation_transform=None
    )

    adele_dataset.return_names = True

    experiment_name = config_handler.read('mlflow', 'experiment_name')
    if experiment_name == "None":
        experiment_name = model_name

    tracking_experiment(
        model,
        train_dataset, val_dataset,
        config_handler,
        experiment_name=experiment_name,
        adele_dataset=adele_dataset
    )
