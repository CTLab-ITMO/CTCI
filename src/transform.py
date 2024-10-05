from typing import Tuple

import albumentations as albu
import hydra
import numpy as np
from albumentations.pytorch import ToTensorV2
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf
from torch import Tensor


def get_transforms(aug_cfg: DictConfig) -> albu.Compose:
    augmentations = OmegaConf.to_object(aug_cfg)
    return albu.Compose([hydra.utils.instantiate(aug) for aug in augmentations])


def cv_image_to_tensor(img: NDArray[float], normalize: bool = True) -> Tensor:
    ops = [ToTensorV2()]
    if normalize:
        ops.insert(0, albu.Normalize())
    to_tensor = albu.Compose(ops)
    return to_tensor(image=img)['image']


def denormalize(
    img: NDArray[float],
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
    max_value: int = 255,
) -> NDArray[int]:
    denorm = albu.Normalize(
        mean=[-me / st for me, st in zip(mean, std)],  # noqa: WPS221
        std=[1.0 / st for st in std],
        always_apply=True,
        max_pixel_value=1.0,
    )
    denorm_img = denorm(image=img)['image'] * max_value
    return denorm_img.astype(np.uint8)


def tensor_to_cv_image(tensor: Tensor) -> NDArray[float]:
    return tensor.permute(1, 2, 0).cpu().numpy()
