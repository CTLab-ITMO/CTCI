# these are the augmentations which we recommend to use while training.
# they are initialized when the augmentations flag in the config file is True.
# you can also modify these augmentations by the config.

import albumentations as albu


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

):
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


def get_resize(height, width):
    return albu.Resize(height=height, width=width)


def get_resize_from_config(config):
    return get_resize(config.read('dataset', 'image_size', 'height'), config.read('dataset', 'image_size', 'width'))


def get_augmentations_from_config(config):
    return get_augmentations(**config.read('dataset', 'augmentation'))
