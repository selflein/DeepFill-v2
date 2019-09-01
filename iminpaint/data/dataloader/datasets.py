from typing import List
from pathlib import Path

from skimage import io
from torch.utils.data import Dataset

from iminpaint.data.dataloader.data_utils import generate_mask


class InpaintingDataset(Dataset):
    def __init__(self, img_paths: List[Path], transforms=None):
        super(InpaintingDataset).__init__()
        self.img_paths = img_paths
        self.transforms = transforms

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, i):
        img = io.imread(str(self.img_paths[i]))

        if self.transforms:
            img = self.transforms(img)

        mask = generate_mask(
            range_vertices=(2, 5),
            range_length=(10, 50),
            range_brush_width=(10, 50),
            range_angle=(0, 1),
            image_height=img.shape(0),
            image_width=img.shape(1)
        )
        
        return img, mask
