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
import torch.distributed as dist

# i2t imports
from i2t.utils import instantiate


__all__ = ['I2T']


logger = logging.getLogger(__name__)


class I2T(pl.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.save_hyperparameters(config)
        self.modalities = ('text', 'image')
        self.encoders = nn.ModuleDict({
            modality: instantiate(self.hparams.model.get(modality))
            for modality in self.modalities
        })
        # hard-coded ntxent loss for simplicity
        self.loss = nn.CrossEntropyLoss()

    def forward(self, batch: Dict) -> Dict:
        return {
            modality: self.encoders[modality](batch[modality])
            for modality in self.modalities
        }

    def training_step(self, batch: Dict, batch_idx: int) -> Dict:
        return self.step_model(self(batch), mode='train')

    def validation_step(self, batch: Dict, batch_idx: int) -> Dict:
        return self.step_model(self(batch), mode='val')

    def step_model(self, local_outputs: Dict[str, torch.Tensor], mode: str) -> Dict:
        logits = self.gather_logits(local_outputs) / self.hparams.loss.temperature
        losses = self.calculate_loss(logits)
        metrics = self.calculate_metrics(logits)
        self.log_dict({f'{mode}/{name}': value for name, value in losses.items()})
        self.log_dict({f'{mode}/{name}': value for name, value in metrics.items()})
        return {'loss': losses['nce']}

    def gather_logits(self, local_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Calculate logits for globa batch gathered from all devices.

        Uses a trick to reverse gradient flow,
        see https://github.com/KevinMusgrave/pytorch-metric-learning/issues/10#issuecomment-593170720
        """
        gathered_outputs = {
            key: [torch.ones_like(value) for _ in range(dist.get_world_size())]
            for key, value in local_outputs.items()
        }
        for key, tensor_list in gathered_outputs.items():
            dist.all_gather(tensor_list, local_outputs[key])
            tensor_list[dist.get_rank()] = local_outputs[key]

        image_features = torch.cat(gathered_outputs['image'], dim=0)
        text_features = torch.cat(gathered_outputs['text'], dim=0)
        return image_features @ text_features.T

    def calculate_loss(self, logits: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Contrastive NCE loss, see https://paperswithcode.com/method/nt-xent for details
        """
        labels = torch.arange(0, logits.shape[0], device=self.device)
        loss_i2t = self.loss(logits, labels)
        loss_t2i = self.loss(logits.T, labels)
        loss = (loss_i2t + loss_t2i) / 2
        return {
            'nce_i2t': loss_i2t,
            'nce_t2i': loss_t2i,
            'nce': loss
        }

    def calculate_metrics(self, logits: torch.Tensor) -> Dict[str, torch.Tensor]:
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
