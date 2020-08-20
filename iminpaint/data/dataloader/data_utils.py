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
        transforms.Normalize((0.4422, 0.4462, 0.4205), (0.2678, 0.2600, 0.2976))
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


def generate_mask(range_vertices: Tuple[int, int],
                  range_length: Tuple[float, float],
                  range_brush_width: Tuple[float, float],
                  range_angle: Tuple[float, float],
                  range_num_patches: Tuple[int, int],
                  image_width: int = 256,
                  image_height: int = 256,
                  ) -> torch.Tensor:
    """
    Generates random mask based on parameters given to be blacked out in the
    image for inpainting.

    Args:
        range_vertices: Tuple of min. and max. number of vertices of the line.
        range_length: Tuple of min. and max. length of each line segment between
            vertices.
        range_brush_width: Tuple of min. and max. width of the brush of each
            line segment.
        range_angle: Tuple of min. and max. turning angle of brush in radiants.
        num_patches: Number of times the stroking algorithm is run.
        image_width: Width of the image in pixels.
        image_height: Height of the image in pixels.

    Returns:
        mask: A mask of a single line generated according to the given
            parameters.
    """
    mask = np.zeros((image_height, image_width))
    num_patches = random.randint(*range_num_patches)

    for i in range(num_patches):
        num_vertex = random.randint(*range_vertices)

        start_x = random.randint(0, image_width)
        start_y = random.randint(0, image_height)

        for i in range(num_vertex):
            angle = random.uniform(*range_angle)

            if i % 2 == 0:
                angle = 2 * np.pi - angle

            length = random.uniform(*range_length)
            brush_width = random.randint(*range_brush_width)

            end_x = int(start_x + length * np.sin(angle))
            end_y = int(start_y + length * np.cos(angle))

            cv2.line(mask, (start_x, start_y), (end_x, end_y), color=1,
                    thickness=brush_width)
            cv2.circle(mask, (end_x, end_y), radius=int(brush_width/2),
                    thickness=-1, color=1)

            start_x = end_x
            start_y = end_y

    if random.random() < 0.5:
        mask = np.fliplr(mask)

    if random.random() < 0.5:
        mask = np.flipud(mask)

    return mask.astype(np.int8)
