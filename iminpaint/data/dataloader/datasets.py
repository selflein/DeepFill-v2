from typing import List
from pathlib import Path

from torch.utils.data import Dataset


class InpaintingDataset(Dataset):
    def __init__(self, img_paths: List[Path]):
        super(InpaintingDataset).__init__()
        self.img_paths = img_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, i):
        return self.img_paths[i]
