from argparse import ArgumentParser

from tqdm import tqdm
import pandas as pd
from flickrapi import FlickrAPI

key = ''
secret = ''

parser = ArgumentParser(description='Download images from Flickr api with the respective tag.')
parser.add_argument('--max_count', required=True)
parser.add_argument('--tag', required=True)


def get_urls(image_tag, max_count):
    flickr = FlickrAPI(key, secret)
    photos = flickr.walk(
        text=image_tag,
        tag_mode='all',
        tags=image_tag,
        extras='url_o',
        per_page=50,
        sort='relevance'
    )

    urls = []
    for count, photo in tqdm(enumerate(photos), ):
        if count < max_count:
            print("Fetching url for image number {}".format(count))
            try:
                urls.append(photo.get('url_o'))
            except:
                print("URL for image number {} could not be fetched".format(count))

    urls = pd.Series(urls)
    print("Writing out the urls in the current directory")
    urls.to_csv(image_tag + "_urls.csv")


if __name__=='__main__':
    args = parser.parse_args()

    get_urls(args['tag'], args['max_count'])
