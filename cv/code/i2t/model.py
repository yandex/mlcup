# generic imports
from typing import Dict, Optional

# torch imports
import torch
from torch import nn

# custom imports
from bpemb import BPEmb
from torchvision.models.resnet import ResNet
from segmentation_models_pytorch.encoders import get_encoder


class ImageModel(nn.Module):
    def __init__(
        self,
        encoder_name: str = 'resnet50',
        weights: Optional[str] = 'imagenet',
    ):
        super().__init__()
        self.encoder = get_encoder(name=encoder_name, weights=weights)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.output_dim = self.encoder.out_channels[-1]

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.encoder(image)[-1]
        x = self.avgpool(x)
        return torch.flatten(x, start_dim=1)


class TextModel(nn.Module):
    def __init__(
        self,
        hidden_size: int = 200,
        hidden_layers: int = 3,
        embedding_size: int = 200,
        vocab_size: int = 200000,
        pretrained_embeddings: bool = True,
        freeze_embeddings: bool = False
    ):
        super().__init__()

        if pretrained_embeddings:
            emb = BPEmb(lang="ru", dim=embedding_size, vs=vocab_size)
            self.embedding = nn.EmbeddingBag.from_pretrained(
                torch.tensor(emb.vectors),
                freeze=freeze_embeddings,
                sparse=False,
            )
        else:
            self.embedding = nn.EmbeddingBag(
                vocab_size,
                embedding_dim=embedding_size,
                sparse=False
            )

        self.output_dim = hidden_size

        in_channels = [embedding_size, *(hidden_size for _ in range(hidden_layers))]
        out_channels = [hidden_size for _ in range(hidden_layers + 1)]
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(inc, outc),
                nn.BatchNorm1d(outc),
                nn.ReLU()
            )
            for inc, outc in zip(in_channels, out_channels)
        ])

    def forward(self, text_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.embedding(text_data['ids'], text_data['offsets'])
        for block in self.blocks:
            x = block(x)
        return x
