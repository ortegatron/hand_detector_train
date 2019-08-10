from enum import Enum

import tensorflow as tf
import cv2


regularizer_conv = 0.004
regularizer_dsconv = 0.0004
batchnorm_fused = True
activation_fn = tf.nn.relu


def read_imgfile(path, width=None, height=None):
    val_image = cv2.imread(path, cv2.IMREAD_COLOR)
    if width is not None and height is not None:
        val_image = cv2.resize(val_image, (width, height))
    return val_image


def get_sample_images(w, h):
    val_image = [
        read_imgfile('images/hand_sample.png', w, h),
        read_imgfile('images/hand_synth_sample1.jpg', w, h),
        read_imgfile('images/hand_synth_sample2.jpg', w, h),
        read_imgfile('images/hand_synth_sample3.jpg', w, h),
    ]
    return val_image


def to_str(s):
    if not isinstance(s, str):
        return s.decode('utf-8')
    return s
