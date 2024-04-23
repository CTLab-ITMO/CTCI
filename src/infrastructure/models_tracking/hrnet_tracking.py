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

import timm

from src.features.segmentation.dataset import get_transform_by_config, get_train_dataset_by_config, get_val_dataset_by_config
from src.models.hrnet.model import HRNetModel
from src.infrastructure.tracking import tracking_experiment
from src.models.utils.config import read_yaml_config

sys.path.append(f"../src/")


if __name__ == "__main__":
    config_path = sys.argv[1]
    config_handler = read_yaml_config(config_path)

    model_name = config_handler.read('model', 'model_name')
    net = timm.create_model(model_name, features_only=True, pretrained=True)
    model = HRNetModel(net=net)

    transform = get_transform_by_config(config_handler)
    train_dataset = get_train_dataset_by_config(config_handler, transform)
    val_dataset = get_val_dataset_by_config(config_handler, transform)

    experiment_name = config_handler.read('mlflow', 'experiment_name')
    if experiment_name == "None":
        experiment_name = model_name

    tracking_experiment(
        model,
        train_dataset, val_dataset,
        config_handler,
        experiment_name=experiment_name
    )

