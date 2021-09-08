# generic imports
from typing import Any, Callable, Dict
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
from torchnlp.encoders.text import stack_and_pad_tensors
from torch.utils.data._utils.collate import default_collate


logger = logging.getLogger(__name__)


__all__ = ['I2TDataset']



class I2TDataset(Dataset):
    def __init__(
        self,
        metadata_file: Path,
        metadata_slice: slice,
        images_directory: Path,
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
        self.data = self.data[metadata_slice]

        self.images_directory = Path(images_directory)
        self.randomize = randomize
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        if randomize:
            self.image_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            self.image_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ])
        self.tokenizer = BPEmb(lang="ru", dim=200, vs=200000, segmentation_only=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img, queries = self.data[idx]
        img = Image.open((self.images_directory / str(img)).with_suffix('.jpg'))
        img = img.convert('RGB')
        img = self.image_transform(img)
        if self.randomize:
            query = np.random.choice(queries)
        else:
            query = self.queries[0]
        return {'image': img, 'text': self.tokenizer.encode_ids(query)}

    @staticmethod
    def collate_fn(items):
        ids = []
        offsets = [0]
        for item in items:
            ids.append(torch.tensor(item['text'], dtype=torch.int64))
            offsets.append(len(item['text']))
        return {
            'image': default_collate([x['image'] for x in items]),
            'text': {
                'ids': torch.cat(ids),
                'offsets': torch.tensor(offsets[:-1]).cumsum(dim=0)
            }
        }


def get_dataloaders(
    metadata_file: str,
    images_directory: str,
    batch_size: int = 512,
    dataloader_workers: int = 8,
    num_train_samples: int = 4e6,
    tqdm_load: bool = False
):
    train_dataset = I2TDataset(
        metadata_file=Path(metadata_file),
        metadata_slice=slice(0, num_train_samples),
        images_directory=Path(images_directory),
        randomize=True,
        tqdm_load=tqdm_load
    )
    val_dataset = I2TDataset(
        metadata_file=Path(metadata_file),
        metadata_slice=slice(num_train_samples, None),
        images_directory=Path(images_directory),
        randomize=False,
        tqdm_load=tqdm_load
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=I2TDataset.collate_fn,
        shuffle=True,
        num_workers=dataloader_workers,
        drop_last=True
    )
    val_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=I2TDataset.collate_fn,
        shuffle=False,
        num_workers=dataloader_workers,
        drop_last=False
    )

    return train_dataloader, val_dataloader
