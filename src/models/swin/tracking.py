import sys
import os.path as osp
sys.path.append('..')

from transformers import AutoImageProcessor, Swinv2Model

from src.features.segmentation.transformers_dataset import SegmentationDataset
from src.models.swin.model import Swin
from src.models.tracking import tracking_experiment
from src.models.utils.config import read_yaml_config

if __name__ == "__main__":
    config_path = sys.argv[1]
    config_data = read_yaml_config(config_path)

    model_str = rf"microsoft/{config_data['model']['model_name']}-{config_data['model']['model_type']}"
    image_processor = AutoImageProcessor.from_pretrained(model_str)
    net = Swinv2Model.from_pretrained(model_str)

    model = Swin(net=net, image_processor=image_processor)

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
        experiment_name="swin-test"
    )