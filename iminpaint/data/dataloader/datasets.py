from typing import List
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from iminpaint.data.dataloader.data_utils import generate_freeform_mask, generate_rect_masks


class InpaintingDataset(Dataset):
    def __init__(self, 
                 img_paths: List[Path], 
                 edge_masks_folder: Path,
                 freeform_mask: bool = True,
                 rect_mask: bool = True,
                 transforms=ToTensor()):
        super(InpaintingDataset).__init__()
        self.img_paths = img_paths
        self.transforms = transforms
        self.edge_masks_folder = edge_masks_folder
        self.freeform_mask = freeform_mask
        self.rect_mask = rect_mask

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, i):
        img_path = self.img_paths[i]
        img = Image.open(str(img_path))

        if self.transforms:
            img = self.transforms(img)

        c, h, w = img.shape
        mask = np.ones((h, w), np.float32)
        if self.freeform_mask:
            freeform_mask = generate_freeform_mask(
                mask_dims=(h, w),
                range_vertices=(2, 10),
                range_brush_width=(10, 40),
                angle_range=np.pi / 15,
                range_num_patches=(1, 4),
            )
            mask *= freeform_mask

        if self.rect_mask:
            rect_mask = generate_rect_masks(
                mask_dims=(h, w),
                range_num_rects=(0, 4),
                range_size=(int(.1 * min(h, w)), int(0.3 * min(h, w)))
            )
            mask *= rect_mask

        mask = torch.from_numpy(mask).unsqueeze(0)
        masked_img = img * mask

        edge_mask_path = str(self.edge_masks_folder / img_path.name)
        edges_mask = torch.from_numpy(
            (np.array(Image.open(edge_mask_path)) / 255.).astype(np.float32)
        ).unsqueeze(0)
        # Only draw in masked regions
        edges_mask[mask == 1] = 0

        return img, masked_img, mask, edges_mask
