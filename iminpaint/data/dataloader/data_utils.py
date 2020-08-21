import random
from typing import Tuple

import cv2
import torch
import numpy as np
from torchvision import transforms

from tqdm import tqdm


def regular_data_transform():
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4422, 0.4462, 0.4205), (0.2678, 0.2600, 0.2976))
    ])
    return transform


def online_mean_and_sd(loader):
    """Compute the mean and std in an online fashion using
    Var[x] = E[X^2] - E^2[X].

    Args:
        loader: DataLoader instance with ImageFolder dataset.

    Returns:
        Tuple (mean_r, mean_g, mean_b) and (std_r, std_g, std_b),
        i.e. the mean and std of every image channel.
    """
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for data in tqdm(loader, total=len(loader)):
        data = data[0]
        b, c, h, w = data.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(data, dim=[0, 2, 3])
        sum_of_square = torch.sum(data ** 2, dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

        cnt += nb_pixels

    return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)


def generate_freeform_mask(
        mask_dims: Tuple[int, int],
        range_vertices: Tuple[int, int],
        range_brush_width: Tuple[float, float],
        range_num_patches: Tuple[int, int],
        angle_range: float,
) -> np.array:
    """
    Generates random mask based on parameters given to be blacked out in the
    image for inpainting.

    Args:
        mask_dims: Output mask dimensions.
        range_vertices: Tuple of min. and max. number of vertices of the line.
        range_brush_width: Tuple of min. and max. width of the brush of each
            line segment.
        angle_range: Range of angle of strokes.
        range_num_patches: Number of times the stroking algorithm is run.

    Returns:
        mask: Numpy array of shape `mask_dims` where masked locations contain 1
         and unmasked 0.
    """
    h, w = mask_dims
    mask = np.zeros(mask_dims, np.float32)
    num_patches = random.randint(*range_num_patches)

    average_radius = np.sqrt(w * w + h * h) / 8
    mean_angle = 2 * np.pi / 5

    for i in range(num_patches):
        num_vertex = random.randint(*range_vertices)
        brush_width = random.randint(*range_brush_width)

        start_x = random.randint(0, w)
        start_y = random.randint(0, h)
        angle_min = mean_angle - random.uniform(0, angle_range)
        angle_max = mean_angle + random.uniform(0, angle_range)

        for j in range(num_vertex):
            angle = random.uniform(angle_min, angle_max)

            if j % 2 == 0:
                angle = 2 * np.pi - angle

            length = np.clip(
                np.random.normal(average_radius, scale=average_radius // 2),
                a_min=0,
                a_max=2 * average_radius
            )

            end_x = np.clip(int(start_x + length * np.sin(angle)), 0, w)
            end_y = np.clip(int(start_y + length * np.cos(angle)), 0, h)

            cv2.line(mask, (start_x, start_y), (end_x, end_y), color=1,
                     thickness=brush_width)
            cv2.circle(mask, (end_x, end_y), radius=int(brush_width / 2),
                       thickness=-1, color=1)

            start_x = end_x
            start_y = end_y

    if random.random() < 0.5:
        mask = np.fliplr(mask)

    if random.random() < 0.5:
        mask = np.flipud(mask)

    return 1 - mask


def generate_rect_masks(
        mask_dims: Tuple[int, int],
        range_num_rects: Tuple[int, int],
        range_size: Tuple[int, int]
) -> np.array:
    """Randomly generate rectangular masked regions within the image.

    Args:
        mask_dims: Output mask dimensions.
        range_num_rects: Range for number of rectangular masked regions to draw.
        range_size: Range for size of each box.

    Returns:
        mask: Numpy array of shape `mask_dims` where masked locations contain 1
         and unmasked 0.
    """
    h, w = mask_dims
    mask = np.ones(mask_dims, np.float32)

    for _ in range(random.randint(*range_num_rects)):
        box_height = random.randint(*range_size)
        box_width = random.randint(*range_size)

        pos_x = random.randint(0, w - box_width - 1)
        pos_y = random.randint(0, h - box_height - 1)

        mask[pos_y: pos_y + box_height, pos_x: pos_x + box_width] = 0.

    return mask
