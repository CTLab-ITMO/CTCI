import sys
import os
import torchvision.transforms as tf

sys.path.append(f"../src/")

import os.path as osp


from src.features.segmentation.dataset import SegmentationDataset
from src.models.hrnet.model import HRNet
from src.infrastructure.tracking import tracking_experiment
from src.models.utils.config import read_yaml_config


if __name__ == "__main__":
    config_path = sys.argv[1]
    config_data = read_yaml_config(config_path)

    model_name = config_data['model']['model_name']
    model = HRNet(freeze_backbone=False)

    tr = tf.Resize((config_data['dataset']['image_size']['height'], config_data['dataset']['image_size']['width']))

    # TODO: create a func to init datasets from config_data
    train_dataset = SegmentationDataset(
        images_dir=osp.join(config_data['dataset']['training_dataset_dirs'][0], "images"),
        masks_dir=osp.join(config_data['dataset']['training_dataset_dirs'][0], "masks"),
        image_transform=tr,
        mask_transform=tr

    )
    val_dataset = SegmentationDataset(
        images_dir=osp.join(config_data['dataset']['validation_dataset_dirs'][0], "images"),
        masks_dir=osp.join(config_data['dataset']['validation_dataset_dirs'][0], "masks"),
        image_transform=tr,
        mask_transform=tr
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

