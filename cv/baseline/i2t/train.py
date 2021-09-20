# typing imports
from omegaconf import DictConfig

# generic imports
import logging
import os

# torch imports
import pytorch_lightning as pl

# custom imports
from i2t.system import I2T
from i2t.data import get_dataloaders
from i2t.utils import instantiate


logger = logging.getLogger(__name__)


def train(cfg: DictConfig) -> None:
    # BUG: pytorch lightning fails on non-existent checkpoint
    resume_from_checkpoint = cfg.train.resume_from_checkpoint
    if (resume_from_checkpoint is not None) and (not os.path.exists(resume_from_checkpoint)):
        logger.warning(f"Not using missing checkpoint {resume_from_checkpoint}, starting from scratch...")
        resume_from_checkpoint = None

    callbacks = [instantiate(x) for x in cfg.train.callbacks.values()]
    plugins = [instantiate(x) for x in cfg.train.plugins.values()]
    trainer = pl.Trainer(
        **cfg.train.trainer_params,
        plugins=plugins,
        callbacks=callbacks,
        logger=instantiate(cfg.train.logger),
        profiler="simple",
        resume_from_checkpoint=resume_from_checkpoint
    )

    model = I2T(config=cfg)
    if resume_from_checkpoint is not None:
        logger.info(f"Resuming training from checkpoint {resume_from_checkpoint}")
    else:
        model.init_pretrain_modules()

    train_dataloader, val_dataloader = get_dataloaders(
        **cfg.data,
        batch_size=cfg.train.batch_size,
    )

    trainer.fit(train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, model=model)
