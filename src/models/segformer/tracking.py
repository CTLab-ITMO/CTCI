import sys
import os.path as osp

import transformers

from src.features.segmentation.transformers_dataset import SegmentationDataset
from src.models.segformer.model import SegFormer
from src.models.tracking import tracking_experiment
from src.models.utils.config import read_yaml_config


if __name__ == "__main__":
    config_path = sys.argv[1]
    config_data = read_yaml_config(config_path)

    image_processor = transformers.SegformerImageProcessor()
    net = transformers.SegformerForSemanticSegmentation.from_pretrained(
        f"nvidia/{config_data['model']['model_name']}-{config_data['model']['model_type']}-finetuned-ade-512-512"
    )
    model = SegFormer(net=net, image_processor=image_processor, device=config_data['model']['device'])

    train_dataset = SegmentationDataset(
        images_dir=osp.join(config_data['dataset']['training_dataset_dirs'][0], "images"),
        masks_dir=osp.join(config_data['dataset']['training_dataset_dirs'][0], "masks")
    )
    val_dataset = SegmentationDataset(
        images_dir=osp.join(config_data['dataset']['validation_dataset_dirs'][0], "images"),
        masks_dir=osp.join(config_data['dataset']['validation_dataset_dirs'][0], "masks")
    )

    tracking_experiment(
        model,
        train_dataset, val_dataset,
        config_data,
        experiment_name="segformer-test"
    )

