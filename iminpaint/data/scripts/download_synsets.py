import os
from pathlib import Path

from iminpaint.data.scripts.utils import download_img

import requests
from tqdm import tqdm

api_url = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid={}'


def download_synset(synset_id: str, download_path: Path):
    with requests.get(api_url.format(synset_id)) as img_urls:
        img_urls = [url.strip() for url in img_urls.text.split('\n')]

        try:
            download_path.mkdir()
        except FileExistsError:
            # check if number of files the same as number of urls
            num_files = len([f for f in download_path.iterdir()])
            if num_files != len(img_urls):
                pass
            else:
                return num_files

        for i, url in tqdm(enumerate(img_urls), desc='Downloading synset {}...'.format(synset_id)):
            try:  
                download_img(url, download_path / (str(i) + '.png'))
            except:
                print('Error when scripts {}.'.format(url))

    return len([f for f in download_path.iterdir()])


if __name__ == '__main__':
    cur_path = Path(os.path.dirname(os.path.realpath(__file__)))
    syn_ids_file = cur_path / 'imagenet_wordnet_ids.txt'

    num_imgs = 0
    with syn_ids_file.open() as syn_ids:
        for syn_id in syn_ids:
            syn_id = syn_id.strip()
            num_imgs += download_synset(syn_id.strip(), cur_path / 'imagenet' / syn_id)
    
    print('Total number of images: {}'.format(num_imgs))