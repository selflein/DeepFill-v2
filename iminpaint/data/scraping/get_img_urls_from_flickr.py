import os
from pathlib import Path
from argparse import ArgumentParser

from tqdm import tqdm
import pandas as pd
from flickrapi import FlickrAPI

key = os.environ['flickr_api_key']
secret = os.environ['flickr_api_secret']


def get_urls(search_text, image_tags, max_count):
    flickr = FlickrAPI(key, secret)

    # API documentation: https://www.flickr.com/services/api/flickr.photos.search.html
    photos = flickr.walk(
        text=search_text,
        tag_mode='any',
        tags=image_tags,
        extras='url_m',  # url_o for original-res images (others: s, m, l)
        per_page=500,
        sort='relevance',
        content_type=1,
        media='photos'
    )

    urls = []
    with tqdm(total=max_count) as pbar:
        for count, photo in enumerate(photos):
            if count < max_count:
                try:
                    urls.append(photo.get('url_m'))
                except Exception:
                    pbar.write("URL for image number {} could not be fetched".format(count))
            else:
                break
            pbar.update()

    return urls


def write_to_disk(urls, path: Path):
    pd.DataFrame(urls).to_csv(path, index=False, header=False)
    print('Writing URLs to "{}".'.format(str(path)))


if __name__ == '__main__':
    parser = ArgumentParser(description='Download image urls from Flickr api with the respective tag.')
    parser.add_argument('--max_count', required=True)
    parser.add_argument('--tags', required=True, help='Comma delimited list of tags')
    parser.add_argument('--text', default=' ')
    parser.add_argument('--save_path', required=True)

    args = parser.parse_args()
    img_tags = args.tags
    save_folder = Path(args.save_path)

    url_list = get_urls(args.text, img_tags, int(args.max_count))
    write_to_disk(url_list, save_folder / img_tags.replace(',', '_') + "_urls.csv")
