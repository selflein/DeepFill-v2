from typing import List
from pathlib import Path

import numpy as np
from skimage import io
from torch.utils.data import Dataset

from iminpaint.data.dataloader.data_utils import generate_mask


class InpaintingDataset(Dataset):
    def __init__(self, 
                 img_paths: List[Path], 
                 edge_masks_folder=Path('data/datasets/flickr_dataset/training_imgs_edges'), 
                 transforms=None):
        super(InpaintingDataset).__init__()
        self.img_paths = img_paths
        self.transforms = transforms
        self.edge_masks_folder = edge_masks_folder
        io.use_plugin('pil')

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, i):
        img_path = self.img_paths[i]
        img = io.imread(str(img_path))

        if self.transforms:
            img = self.transforms(img)
        
        edge_mask_path = self.edge_masks_folder / img_path.name
        edges_mask = (io.imread(edge_mask_path) / 255).astype(np.int8)

        draw_mask = generate_mask(
            range_vertices=(2, 10),
            range_length=(20, 50),
            range_brush_width=(20, 50),
            range_angle=(-np.pi, np.pi),
            range_num_patches=(1, 3),
            image_height=img.shape[0],
            image_width=img.shape[1]
        )
        
        return img, draw_mask, edges_mask
