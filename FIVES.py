import os
import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from typing import List, Callable, Optional
import torch

class FIVESDataset(Dataset):
    def __init__(
    self,
    img_paths: List[str],
    mask_paths: List[str],
    transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
    ):
        self.transform = transform or T.ToTensor()
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')
        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)
        mask = (mask > 0).float() # type: ignore
        return img, mask
