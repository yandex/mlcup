# typing imports
from typing import List, Dict, Tuple
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.optim import Optimizer
from omegaconf import DictConfig

# generic imports
import logging
from omegaconf import OmegaConf

# torch imports
import pytorch_lightning as pl
import torch
from torch import nn

# i2t imports
from i2t.utils import instantiate


__all__ = ['I2T']


logger = logging.getLogger(__name__)


class I2T(pl.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.save_hyperparameters(config)

        self.image_encoder = instantiate(self.hparams.model.image)
        self.image_projector = nn.Linear(self.image_encoder.output_dim, self.hparams.model.joint_dim)
        self.text_encoder = instantiate(self.hparams.model.text)
        self.text_projector = nn.Linear(self.text_encoder.output_dim, self.hparams.model.joint_dim)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, batch: Dict) -> Dict:
        image_features = nn.functional.normalize(self.image_projector(self.image_encoder(batch['image'])))
        text_features = nn.functional.normalize(self.text_projector(self.text_encoder(batch['text'])))
        return {
            'image_features': image_features,
            'text_features':text_features,
            'logits': image_features @ text_features.T
        }

    def training_step(self, batch: Dict, batch_idx: int) -> Dict:
        return self.step_model(batch, mode='train')

    def validation_step(self, batch: Dict, batch_idx: int) -> Dict:
        return self.step_model(batch, mode='val')

    def step_model(self, batch: Dict, mode: str) -> Dict:
        predict = self(batch)
        logits = predict['logits'] / self.hparams.loss.temperature
        losses = self.calculate_loss(logits)
        metrics = self.calculate_metrics(logits)
        self.log_dict({f'{mode}/{name}': value for name, value in losses.items()})
        self.log_dict({f'{mode}/{name}': value for name, value in metrics.items()})
        return {'loss': losses['nce']}

    def calculate_loss(self, logits):
        labels = torch.arange(0, logits.shape[0], device=self.device)
        loss_i2t = self.loss(logits, labels)
        loss_t2i = self.loss(logits.T, labels)
        loss = (loss_i2t + loss_t2i) / 2
        return {
            'nce_i2t': loss_i2t,
            'nce_t2i': loss_t2i,
            'nce': loss
        }

    def calculate_metrics(self, logits):
        return {
            'binary_accuracy': (logits.diag().unsqueeze(1) >= logits).to(dtype=torch.float32).mean()
        }

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[LRScheduler]]:
        config = self.hparams.optimization
        optimizer = instantiate(config.optimizer, params=self.parameters())
        lr_scheduler = OmegaConf.to_container(config.lr_scheduler, resolve=True)
        lr_scheduler['scheduler'] = instantiate(lr_scheduler['scheduler'], optimizer)

        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler
        }
