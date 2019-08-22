from pathlib import Path
from functools import partial
from argparse import ArgumentParser
from multiprocessing import Pool

from tqdm import tqdm
import pandas as pd

from iminpaint.data.scraping.utils import download_img


def download_images_threaded(img_urls, save_path: Path, num_threads: int = 4):
    start_img_download_infer_name_from_url_partial = partial(
        start_img_download_infer_name_from_url, save_folder=save_path)

    downloader_threads = Pool(processes=num_threads)
    downloader_threads.map(start_img_download_infer_name_from_url_partial, img_urls)
    downloader_threads.close()
    downloader_threads.join()


def start_img_download_infer_name_from_url(url: str, save_folder: Path):
    print('Downloading image: "{}"'.format(url))
    url_components = url.split('/')
    img_name = url_components[-2] + '_' + url_components[-1].split('.')[0] + '.png'
    download_img(url, save_folder / img_name)


def load_urls_from_csv(csv_path: Path):
    img_urls = pd.read_csv(csv_path, header=None, names=['URL'])
    return img_urls['URL'].tolist()


def download_images(img_urls, save_folder: Path):
    existing_imgs = set([p for p in save_folder.iterdir() if p.is_file()])

    for url in tqdm(img_urls, desc='Downloading images'):
        try:
            url_components = url.split('/')
            img_name = url_components[-2] + '_' + url_components[-1].split('.')[0] + '.png'
            img_path = save_folder / img_name
        except Exception:
            continue

        if img_path not in existing_imgs:
            try:
                download_img(url, img_path)
            except Exception as e:
                print(e)


if __name__ == '__main__':
    parser = ArgumentParser(description='Download images from URLs saved in a CSV')
    parser.add_argument('--csv_path', required=True)
    parser.add_argument('--save_path', required=True)
    parser.add_argument('--num_threads')

    args = parser.parse_args()
    urls = load_urls_from_csv(Path(args.csv_path))

    if args.num_threads:
        download_images_threaded(urls, Path(args.save_path), int(args.num_threads))
    else:
        download_images(urls, Path(args.save_path))
