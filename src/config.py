import typing as tp

from omegaconf import OmegaConf, DictConfig
from pydantic import BaseModel, ConfigDict


class _BaseValidatedConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')


class LossConfig(_BaseValidatedConfig):
    name: str
    weight: float
    loss_fn: str


class AugmentationConfig(_BaseValidatedConfig):
    train: tp.Optional[DictConfig] = None
    valid: tp.Optional[DictConfig] = None


class DataConfig(_BaseValidatedConfig):
    img_size: tp.List[int] = [224, 224]
    batch_size: int = 32
    num_workers: int = 6
    pin_memory: bool = True
    num_samples: int = 1024
    data_dir: str = 'data'
    train_folder: str = 'train'
    valid_folder: str = 'val'
    test_folder: tp.Optional[str] = None
    # by default images_dir and masks_dir should have names images and masks


class ModuleConfig(_BaseValidatedConfig):
    model: tp.Dict[str, tp.Any]  # type: ignore
    threshold: float
    num_classes: int

    lr: float
    optimizer: str
    scheduler: str

    cls_losses: tp.List[LossConfig]
    seg_losses: tp.List[LossConfig]


class TrainerConfig(_BaseValidatedConfig):
    fast_dev_run: bool
    accelerator: str
    log_every_n_steps: int
    max_epochs: int
    min_epochs: int


class ExperimentConfig(_BaseValidatedConfig):
    project_name: str
    experiment_name: str

    data_config: DataConfig
    trainer_config: TrainerConfig
    aug_config: tp.Optional[AugmentationConfig]

    @classmethod
    def from_yaml(cls, path: str) -> 'ExperimentConfig':
        cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        return cls(**cfg)
