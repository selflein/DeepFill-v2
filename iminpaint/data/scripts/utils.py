from io import BytesIO
from pathlib import Path

import requests
from PIL import Image


def download_img(url: str, save_path: Path, min_size=(256, 256)):
    with requests.get(url, timeout=2) as resp:
        if resp.status_code == requests.codes.ok:
            img = Image.open(BytesIO(resp.content))
            if all(a >= b for a, b in zip(img.size, min_size)):
                img.save(save_path)
        else:
            print('Error code: {}'.format(resp.status_code))
