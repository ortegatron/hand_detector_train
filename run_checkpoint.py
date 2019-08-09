import argparse
import logging
import os

import tensorflow as tf
from networks import get_network

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.95
config.gpu_options.allow_growth = True


if __name__ == '__main__':
    """
    Use this script to just save graph and checkpoint.
    While training, checkpoints are saved. You can test them with this python code.
    """
    parser = argparse.ArgumentParser(description='Tensorflow Pose Estimation Graph Extractor')
    args = parser.parse_args()

    w = h = None
    input_node = tf.placeholder(tf.float32, shape=(None, h, w, 3), name='image')
    net, pretrain_path, last_layer = get_network("vgg", input_node)

    with tf.Session(config=config) as sess:
        net.load(pretrain_path, sess, True)
        tf.train.write_graph(sess.graph_def, './tmp', 'graph_definition.pb', as_text=True)
        flops = tf.profiler.profile(None, cmd='graph', options=tf.profiler.ProfileOptionBuilder.float_operation())
        print('FLOP = ', flops.total_float_ops / float(1e6))
