import torch
from torchvision import transforms

from tqdm import tqdm


def regular_data_transform():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4576, 0.4412, 0.4081), (0.2689, 0.2668, 0.2849))
    ])
    return transform


def online_mean_and_sd(loader):
    """Compute the mean and std in an online fashion using
        Var[x] = E[X^2] - E^2[X].

        Args:
            loader: DataLoader instance with ImageFolder dataset.

        Returns:
            Tuple (mean_r, mean_g, mean_b) and (std_r, std_g, std_b), i.e.
            the mean and std of every image channel.
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
