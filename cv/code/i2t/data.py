# generic imports
from typing import Any, Optional, Dict
import numpy as np
import logging
import jsonlines
from pathlib import Path
from tqdm.auto import tqdm

# torch imports
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# custom imports
from bpemb import BPEmb
from sentencepiece import SentencePieceProcessor
from torch.utils.data._utils.collate import default_collate
from i2t.utils import instantiate, ClassDescription


logger = logging.getLogger(__name__)


__all__ = ['I2TDataset']


def get_image_transform(randomize: bool):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    if randomize:
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])


def text_collate_fn(items):
    ids = []
    offsets = [0]
    for item in items:
        ids.append(torch.tensor(item, dtype=torch.int64))
        offsets.append(len(item))
    return {
        'ids': torch.cat(ids),
        'offsets': torch.tensor(offsets[:-1]).cumsum(dim=0)
    }



class BPEmbTokenizer(BPEmb):
    def __call__(self, text):
        return self.encode_ids(text)


class SentencePieceTokenizer(SentencePieceProcessor):
    def __call__(self, text):
        return self.encode(text, out_type=int)


class I2TDataset(Dataset):
    def __init__(
        self,
        metadata_file: Path,
        images_directory: Path,
        tokenizer: ClassDescription,
        start: int = 0,
        end: Optional[int] = None,
        randomize: bool = True,
        tqdm_load: bool = False
    ):
        super().__init__()
        self.data = []
        with jsonlines.open(metadata_file) as reader:
            if tqdm_load:
                reader = tqdm(reader)
            for obj in reader:
                self.data.append((obj['image'], obj['queries']))
        self.data = self.data[slice(start, end)]

        self.images_directory = Path(images_directory)
        self.randomize = randomize
        self.image_transform = get_image_transform(randomize=randomize)
        self.tokenizer = instantiate(tokenizer)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img, texts = self.data[idx]
        img = Image.open((self.images_directory / str(img)).with_suffix('.jpg'))
        img = img.convert('RGB')
        img = self.image_transform(img)
        if self.randomize:
            text = np.random.choice(texts)
        else:
            text = texts[0]
        return {'image': img, 'text': self.tokenizer(text)}

    @staticmethod
    def collate_fn(items):
        return {
            'image': default_collate([x['image'] for x in items]),
            'text': text_collate_fn([x['text'] for x in items])
        }


def get_dataloaders(
    train: ClassDescription,
    val: ClassDescription,
    batch_size: int,
    dataloader_workers: int
):
    train_dataset = instantiate(train)
    val_dataset = instantiate(val)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=train_dataset.collate_fn,
        shuffle=True,
        num_workers=dataloader_workers,
        drop_last=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=val_dataset.collate_fn,
        shuffle=False,
        num_workers=dataloader_workers,
        drop_last=False
    )

    return train_dataloader, val_dataloader
