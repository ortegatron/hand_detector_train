import logging
import math
import multiprocessing
import struct
import sys
import threading

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

from contextlib import contextmanager

import os
import random
import requests
import cv2
import numpy as np
import time

import tensorflow as tf

from tensorpack.dataflow import MultiThreadMapData
from tensorpack.dataflow.image import MapDataComponent
from tensorpack.dataflow.common import BatchData, MapData
from tensorpack.dataflow.parallel import PrefetchData
from tensorpack.dataflow.base import RNGDataFlow, DataFlowTerminated

from numba import jit

from dataflow import DataFlowToQueue
from hands_metadata import HandsMetadata
from synthhands import SynthHands

logging.getLogger("requests").setLevel(logging.WARNING)
logger = logging.getLogger('pose_dataset')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

def read_image_url(metas):
    for meta in metas:
        meta.img =  cv2.imread(meta.img_url, cv2.IMREAD_COLOR)
        if meta.img is None:
            logger.warning('image not read, path=%s' % meta.img_url)
            raise Exception()
    return metas

def get_dataflow(path, is_train):
    ds = SynthHands(path, is_train)       # read data from lmdb
    if is_train:
        ds = MapData(ds, read_image_url)
        ds = MapData(ds, pose_to_img)
        ds = PrefetchData(ds, 1000, multiprocessing.cpu_count() * 1)
    else:
        ds = MultiThreadMapData(ds, nr_thread=16, map_func=read_image_url, buffer_size=1000)
        ds = MapData(ds, pose_to_img)
        ds = PrefetchData(ds, 100, multiprocessing.cpu_count() // 4)
    return ds

def _get_dataflow_onlyread(path, is_train):
    ds = SynthHands(path, is_train)  # read data from lmdb
    ds = MapData(ds, read_image_url)
    ds = MapData(ds, pose_to_img)
    # ds = PrefetchData(ds, 1000, multiprocessing.cpu_count() * 4)
    return ds

def get_dataflow_batch(path, is_train, batchsize):
    ds = get_dataflow(path, is_train)
    ds = BatchData(ds, batchsize)
    return ds


_network_w = 368
_network_h = 368
_scale = 8

def pose_to_img(meta_l):
    global _network_w, _network_h, _scale
    return [
        meta_l[0].img.astype(np.float16),
        meta_l[0].get_heatmap(target_size=(_network_w // _scale, _network_h // _scale))
    ]


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''


    # df = get_dataflow('/data/public/rw/coco/annotations', True, '/data/public/rw/coco/')
    df = _get_dataflow_onlyread('/home/marcelo/hands/hand_labels_synth', True)
    # df = get_dataflow('/root/coco/annotations', False, img_path='http://gpu-twg.kakaocdn.net/braincloud/COCO/')

    from tensorpack.dataflow.common import TestDataSpeed
    TestDataSpeed(df).start()

    with tf.Session() as sess:
        df.reset_state()
        t1 = time.time()
        for idx, dp in enumerate(df.get_data()):
            if idx == 0:
                for d in dp:
                    logger.info('%d dp shape={}'.format(d.shape))
            print(time.time() - t1)
            t1 = time.time()
            SynthHands.display_image(dp[0], dp[1].astype(np.float32))
            pass

    logger.info('done')
