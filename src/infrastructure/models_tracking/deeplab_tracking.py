import sys

import segmentation_models_pytorch as smp

from src.features.segmentation.dataset import get_transform_by_config, get_train_dataset_by_config, get_val_dataset_by_config
from src.models.deeplab.model import DeepLab
from src.infrastructure.tracking import tracking_experiment
from src.models.utils.config import read_yaml_config

sys.path.append(f"../src/")


if __name__ == "__main__":
    config_path = sys.argv[1]
    config_handler = read_yaml_config(config_path)

    model_name = config_handler.read('model', 'model_name')
    net = smp.DeepLabV3Plus(encoder_name=model_name)
    model = DeepLab(net=net)

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

