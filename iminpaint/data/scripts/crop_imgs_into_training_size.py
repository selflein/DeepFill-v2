from math import ceil
from pathlib import Path
from argparse import ArgumentParser

import PIL
import numpy as np
from PIL import Image
from tqdm import tqdm


def crop_image_into_tiles(image: Image, tile_size=(256, 256)):
    height, width = image.size
    tile_height, tile_width = tile_size

    height_fill = height / tile_height - .5
    width_fill = width / tile_width - .5

    y_coords = np.linspace(0, width, ceil(width_fill) + 2)[1:-1]
    x_coords = np.linspace(0, height, ceil(height_fill) + 2)[1:-1]
    x_size = int(tile_size[0] / 2)
    y_size = int(tile_size[1] / 2)
    img_crops = []
    for x in x_coords:
        for y in y_coords:
            crp = image.crop((x - x_size,
                              y - y_size,
                              x + x_size,
                              y + y_size))
            img_crops.append(crp)

    return img_crops


if __name__ == '__main__':
    parser = ArgumentParser(description='Crop given images into 256x256 image size.')
    parser.add_argument('--in_path', required=True)
    parser.add_argument('--out_path', required=True)
    args = parser.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)

    img_paths = [p for p in in_path.iterdir()]
    existing_imgs = set([p for p in out_path.iterdir() if p.is_file()])

    for img_path in tqdm(img_paths):
        try:
            img = Image.open(img_path)
        except Exception as e:
            print(e)
            continue

        crops = crop_image_into_tiles(img, (330, 330))

        for i, crop in enumerate(crops):
            crop = crop.resize((256, 256), PIL.Image.LANCZOS)
            crop.save(out_path / (img_path.stem + str(i) + img_path.suffix))
