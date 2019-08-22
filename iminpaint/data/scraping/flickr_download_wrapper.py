from pathlib import Path
from argparse import ArgumentParser

from tqdm import tqdm

from iminpaint.data.scraping.download_flickr_imgs import download_images
from iminpaint.data.scraping.get_img_urls_from_flickr import get_urls


if __name__ == '__main__':
    parser = ArgumentParser(description='Download images from Flickr API with the respective tag.')
    parser.add_argument('--number_per_tag', required=True)
    parser.add_argument('--tags', required=True, help='Comma delimited list of tags')
    parser.add_argument('--text', default=' ')
    parser.add_argument('--save_path', required=True)
    args = parser.parse_args()

    tag_list = args.tags.split(',')
    save_folder = Path(args.save_path)

    if not save_folder.exists():
        save_folder.mkdir()

    for tag in tqdm(tag_list, desc='Going through tag list...'):
        urls = get_urls(args.text, tag, int(args.number_per_tag))
        download_images(urls, save_folder)


