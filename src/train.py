import hydra
import lightning
from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from omegaconf import DictConfig

from src.datamodule import CTCIDataModule
from src.lightning_module import CTCILightningModule
from src.callbacks import (
    ClearMLCallback,
    AdeleCallback,
    create_adele_dataloader
)


@hydra.main(version_base=None, config_path='../configs', config_name='config')
def train(cfg: DictConfig) -> None:
    lightning.seed_everything(0)
    datamodule = CTCIDataModule(cfg.data, cfg.aug)

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.experiment.checkpoint_dir,
        save_top_k=3,
        monitor='valid_f1',
        mode='max',
        every_n_epochs=1,
    )

    callbacks = [
        LearningRateMonitor(logging_interval='step'),
        checkpoint_callback,
    ]

    if cfg.track_in_clearml:
        callbacks.append(ClearMLCallback(cfg))

    if cfg.data_cfg.adele_correction:
        callbacks.append(
            AdeleCallback(
                correction_dataloader=create_adele_dataloader(cfg.data),
                save_dir=cfg.data.adele_dir,
                )
            )

    model = CTCILightningModule(cfg=cfg.module)

    trainer = Trainer(
        **dict(cfg.trainer),
        callbacks=callbacks,
    )
    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule)


if __name__ == '__main__':
    train()