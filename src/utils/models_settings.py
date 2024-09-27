"""
Utility functions for model settings.
"""
import torch

from src.models.base_model import BaseModel


def set_image_processor_to_datasets(model: BaseModel, image_size, datasets: list) -> None:
    """
    Sets the image processor to datasets.

    Args:
        model: Model containing the image processor attribute.
        image_size (dict): Size of the input images.
        datasets (list): List of datasets to set the image processor.

    """
    if hasattr(model, 'image_processor'):
        image_processor = model.image_processor
        image_processor.size = image_size
        for dataset in datasets:
            dataset.image_processor = image_processor
            dataset.image_processor = image_processor


def set_gpu(device_name: str) -> torch.device:
    """
    Sets the GPU device for computation.

    Args:
        device_name (str): Name of the GPU device.

    Returns:
        torch.device: Device for computation.
    """
    if not torch.cuda.is_available() and device_name.split(':')[0] != 'mps' and device_name != "directml":
        device_name = 'cpu'
        print("Couldn't find gpu device. Set cpu as device")

    if device_name.split(':')[0] == "cuda":
        print("Set cuda as device")
        torch.cuda.empty_cache()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.benchmark = True

    device = torch.device(device_name)
    return device
