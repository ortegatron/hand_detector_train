import logging
import math
import multiprocessing
import struct
import sys
import json
from PIL import Image
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
import os
import cv2
import numpy as np
import time
import glob

from numba import jit

from hands_metadata import HandsMetadata

from tensorpack.dataflow.base import RNGDataFlow, DataFlowTerminated

class SynthHands(RNGDataFlow):
    @staticmethod
    def display_image(inp, heatmap, as_numpy=False):

        import matplotlib.pyplot as plt

        fig = plt.figure()
        a = fig.add_subplot(2, 2, 1)
        a.set_title('Image')
        plt.imshow(SynthHands.get_bgimg(inp))

        a = fig.add_subplot(2, 2, 2)
        a.set_title('Heatmap')
        plt.imshow(SynthHands.get_bgimg(inp, target_size=(heatmap.shape[1], heatmap.shape[0])), alpha=0.5)
        tmp = np.amax(heatmap, axis=2)
        plt.imshow(tmp, cmap=plt.cm.gray, alpha=0.5)
        plt.colorbar()

        if not as_numpy:
            plt.show()
        else:
            fig.canvas.draw()
            data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            fig.clear()
            plt.close()
            return data

    @staticmethod
    def get_bgimg(inp, target_size=None):
        inp = cv2.cvtColor(inp.astype(np.uint8), cv2.COLOR_BGR2RGB)
        if target_size:
            inp = cv2.resize(inp, target_size, interpolation=cv2.INTER_AREA)
        return inp

    def __init__(self, path, is_train=True):
        self.is_train = is_train
        self.path = path
        self.idxs = []
        for i in [2,3]:
            synth_idx = "synth" + str(i) + "/"
            path = self.path + "/" + synth_idx
            json_files = [f for f in glob.glob(path + "*.json")]
            # Only saves /synthX/XXX
            self.idxs += [synth_idx  + os.path.basename(j).split(".")[0] for j in json_files]

    def size(self):
        return len(self.idxs)

    def get_data(self):
        idxs = np.arange(self.size())
        if self.is_train:
            self.rng.shuffle(idxs)
        else:
            pass
        for idx in self.idxs:
            json_path = self.path + "/" + idx + ".json"
            img_url = self.path + "/" + idx + ".jpg"

            img_meta = {}
            img_meta['width'] , img_meta['height'] = Image.open(img_url).size
            with open(json_path) as json_file:
                data = json.load(json_file)
                annotation = {}
                annotation['keypoints'] = data['hand_pts']
                meta = HandsMetadata(idx, img_url, img_meta, [annotation], sigma=8.0)
                yield [meta]
