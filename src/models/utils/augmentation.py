"""
This module contains augmentations which we recommend to use while training.
They are initialized when the augmentations flag in the config file is True.
It is also possible to modify these augmentations by the config.

"""

import albumentations as albu

from src.models.utils.config import ConfigHandler


def get_augmentations(
        apply_clahe=True,
        apply_norm=True,
        random_flip=0.5,
        shift_limit=0.05,
        scale_limit=0.05,
        crop_height=256,
        crop_width=256,
        crop_p=0.4,
        rotate_limit=30,
        scale_rotate_p=0.5

) -> albu.Compose:
    """
    Returns a composition of augmentations.

    Args:
        apply_clahe (bool): Whether to apply Contrast Limited Adaptive Histogram Equalization (CLAHE). Default is True.
        apply_norm (bool): Whether to apply normalization. Default is True.
        random_flip (float): Probability of applying random horizontal flip. Default is 0.5.
        shift_limit (float): Maximum shift in either direction for ShiftScaleRotate augmentation. Default is 0.05.
        scale_limit (float): Maximum scaling factor for ShiftScaleRotate augmentation. Default is 0.05.
        crop_height (int): Height of the random crop. Default is 256.
        crop_width (int): Width of the random crop. Default is 256.
        crop_p (float): Probability of applying random crop. Default is 0.4.
        rotate_limit (int): Maximum rotation angle for ShiftScaleRotate augmentation. Default is 30.
        scale_rotate_p (float): Probability of applying ShiftScaleRotate augmentation. Default is 0.5.

    Returns:
        albumentations.Compose: Composition of augmentations.
    """
    return albu.Compose([
        albu.CLAHE(always_apply=apply_clahe),
        albu.Normalize(always_apply=apply_norm),
        albu.Flip(p=random_flip),
        albu.RandomCrop(crop_height, crop_width, p=crop_p),
        albu.ShiftScaleRotate(shift_limit=shift_limit,
                              scale_limit=scale_limit,
                              rotate_limit=rotate_limit,
                              p=scale_rotate_p)
    ])


def get_resize(height: int, width: int) -> albu.Resize:
    """
    Returns an augmentation for resizing images.

    Args:
        height (int): Target height after resizing.
        width (int): Target width after resizing.

    Returns:
        albumentations.Resize: Resize augmentation.
    """
    return albu.Resize(height=height, width=width)


def get_augmentations_from_config(config_handler: ConfigHandler) -> albu.Compose:
    """
    Returns a composition of augmentations based on configuration.

    Args:
        config_handler (ConfigHandler): Configuration handler containing augmentation parameters.

    Returns:
        albumentations.Compose: Composition of augmentations.
    """
    return get_augmentations(**config_handler.read('dataset', 'augmentation'))


def get_resize_from_config(config_handler: ConfigHandler) -> albu.Resize:
    """
    Returns an augmentation for resizing images based on configuration.

    Args:
        config_handler (ConfigHandler): Configuration handler containing image size information.

    Returns:
        albumentations.Resize: Resize augmentation.
    """
    return get_resize(
        config_handler.read('dataset', 'image_size', 'height'),
        config_handler.read('dataset', 'image_size', 'width')
    )
