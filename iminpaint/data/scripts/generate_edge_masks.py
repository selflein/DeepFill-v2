import os
import argparse
from pathlib import Path

import cv2 as cv
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from multiprocessing import Pool
from functools import partial
from skimage.io import imread

# Based on
# https://github.com/opencv/opencv/blob/master/samples/dnn/edge_detection.py

parser = argparse.ArgumentParser(description='Generates edge masks for input '
                                             'images.')
parser.add_argument('--input', help='Path to image folder.')
parser.add_argument('--output', help='Path to output image folder.')
parser.add_argument('--prototxt', help='Path to deploy.prototxt', required=True)
parser.add_argument('--caffemodel',
                    help='Path to hed_pretrained_bsds.caffemodel',
                    required=True)
parser.add_argument('--visualize', type=bool, default=False)


class CropLayer(object):
    def __init__(self, params, blobs):
        self.xstart = 0
        self.xend = 0
        self.ystart = 0
        self.yend = 0

    # Our layer receives two inputs. We need to crop the first input blob to
    # match a shape of the second one (keeping batch size and number of
    # channels)
    def getMemoryShapes(self, inputs):
        inputShape, targetShape = inputs[0], inputs[1]
        batchSize, numChannels = inputShape[0], inputShape[1]
        height, width = targetShape[2], targetShape[3]

        self.ystart = (inputShape[2] - targetShape[2]) // 2
        self.xstart = (inputShape[3] - targetShape[3]) // 2
        self.yend = self.ystart + height
        self.xend = self.xstart + width

        return [[batchSize, numChannels, height, width]]

    def forward(self, inputs):
        return [inputs[0][:, :, self.ystart:self.yend, self.xstart:self.xend]]


def chunks(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


if __name__ == '__main__':
    args = parser.parse_args()
    cv.dnn_registerLayer('Crop', CropLayer)

    # Load the model_parts.
    net = cv.dnn.readNetFromCaffe(args.prototxt, args.caffemodel)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_OPENCL)

    existing_files = set([p.name for p in Path(args.output).iterdir()])
    remaining_files = [p for p in Path(args.input).iterdir()
                       if p.name not in existing_files]

    img_files = list(chunks(remaining_files, 16))
    for img_chunk in tqdm(img_files, desc='Creating edge masks...'):
        frames = list(map(imread, img_chunk))
        height, width, ch = frames[0].shape

        inp = cv.dnn.blobFromImages(frames, scalefactor=1.0,
                                    size=(width, height),
                                    mean=(
                                        104.00698793, 116.66876762,
                                        122.67891434),
                                    swapRB=True, crop=False)
        net.setInput(inp)
        out = net.forward()
        heds = [(hed[0, :, :] > .6).astype(np.int8) * 255 for hed in out]
        # kernel = np.ones((3, 3), np.uint8)
        # img_erosion = cv.erode(hed, kernel, iterations=1)
        if args.visualize:
            fig, ax = plt.subplots(3, 1)
            ax[0].imshow(frames[0])
            ax[1].imshow(heds[0])
            # ax[2].imshow(img_erosion)
            plt.show()

        out_paths = map(lambda p: os.path.join(args.output, p.name), img_chunk)
        for path, hed in zip(out_paths, heds):
            cv.imwrite(path, hed)
