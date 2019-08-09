import os
from os.path import dirname, abspath
import tensorflow as tf
import network_base

from network_cmuhand import CmuHandNetwork


def _get_base_path():
    if not os.environ.get('OPENPOSE_MODEL', ''):
        return './models'
    return os.environ.get('OPENPOSE_MODEL')

def get_network(type, placeholder_input, sess_for_load=None, trainable=True):
    if type=="vgg":
        net = CmuHandNetwork({'image': placeholder_input}, trainable=trainable)
        pretrain_path = 'numpy/openpose_vgg16.npy'
        last_layer = 'Mconv7_stage6_L{aux}'
    else:
        raise Exception('Invalid Model Name.')

    pretrain_path_full = os.path.join(_get_base_path(), pretrain_path)
    if sess_for_load is not None:
        if not os.path.isfile(pretrain_path_full):
            raise Exception('Model file doesn\'t exist, path=%s' % pretrain_path_full)
        net.load(os.path.join(_get_base_path(), pretrain_path), sess_for_load)

    return net, pretrain_path_full, last_layer
