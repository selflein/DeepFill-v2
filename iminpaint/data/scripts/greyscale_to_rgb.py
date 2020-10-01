from pathlib import Path
from argparse import ArgumentParser

import cv2 as cv
from tqdm import tqdm

if __name__ == '__main__':
    parser = ArgumentParser(description='Convert greyscale to RGB images '
                                        'inplace.')
    parser.add_argument('--in_path', required=True)
    args = parser.parse_args()

    in_path = Path(args.in_path)

    img_paths = [p for p in in_path.iterdir()]

    for img_path in tqdm(img_paths):
        try:
            im = cv.imread(str(img_path))
        except (ValueError, SyntaxError) as e:
            img_path.unlink()
            print(img_path)
            continue

        if len(im.shape) != 3:
            rgb_img = cv.cvtColor(im, cv.COLOR_GRAY2BGR)
            try:
                cv.imwrite(img_path, rgb_img)
            except ValueError:
                img_path.unlink()
                print(img_path)
                continue
        elif im.shape[2] == 4:
            im = im[:, :, :3]
            try:
                cv.imwrite(img_path, im)
            except ValueError:
                img_path.unlink()
                print(img_path)
                continue
        elif im.shape[2] != 3:
            print("Wrong number of channels")
            print(img_path)
            img_path.unlink()

