import tensorflow as tf
import sys
import cv2
import os
from sys import platform
import argparse
import matplotlib.pyplot as plt
import numpy as np
from networks import get_network
from synthhands import SynthHands



def read_imgfile(path, width, height):
    img = cv2.imread(path)
    if img.shape[0] != width or img.shape[1] != height:
        raise Exception('Image size must be 368x368!')
    return img.astype(np.float16)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test trained detector')
    parser.add_argument('--graph-path', type=str, default='./models/frozengraph.pb')
    parser.add_argument('--image-path', type=str, default='./images/hand_sample.png')
    args = parser.parse_args()

    img = read_imgfile(args.image_path,368,368)

    with tf.gfile.GFile(args.graph_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    graph = tf.get_default_graph()
    tf.import_graph_def(graph_def, name='CmuHand')

    tf_config= tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    with  tf.Session(config=tf_config) as sess:
        graph = tf.get_default_graph()
        inputs = graph.get_tensor_by_name('CmuHand/image:0')
        out = graph.get_tensor_by_name('CmuHand/Openpose/out:0')
        stage_5 = graph.get_tensor_by_name('CmuHand/Mconv7_stage5/BiasAdd:0')
        stage_4 = graph.get_tensor_by_name('CmuHand/Mconv7_stage4/BiasAdd:0')
        stage_3 = graph.get_tensor_by_name('CmuHand/Mconv7_stage3/BiasAdd:0')
        stage_2 = graph.get_tensor_by_name('CmuHand/Mconv7_stage2/BiasAdd:0')
        stage_1 = graph.get_tensor_by_name('CmuHand/conv6_2_CPM/BiasAdd:0')
        stages_outs = sess.run([stage_1, stage_2, stage_3, stage_4, stage_5, out], feed_dict={
            inputs: [img]
        })
        last_stage = stages_outs[-1][0]

    print("Belief maps for last stage, one for each keypoint plus and additional one for the background")
    fig, ax = plt.subplots(nrows=5, ncols=5)
    index = 0
    for row in ax:
        for col in row:
            col.imshow(last_stage[:,:,index])
            index += 1
            if index >= 22:
                break
    plt.show()

    print("Draws last stage belief maps on top of image")
    test_result = SynthHands.display_image(img, last_stage, as_numpy=True)
    test_result = cv2.cvtColor(test_result, cv2.COLOR_RGB2BGR)
    cv2.imshow("Belief Maps",test_result)
    cv2.waitKey(0)


    print("Draws each stage belief maps")
    fig = plt.figure()
    for i, stage_out in enumerate(stages_outs):
        stage_out= stage_out[0]
        a = fig.add_subplot(3, 2, i+1)
        a.set_title('Stage #{}'.format(i+1))
        plt.imshow(SynthHands.get_bgimg(img, target_size=(stage_out.shape[1], stage_out.shape[0])), alpha=0.5)
        tmp = np.amax(stage_out, axis=2)
        plt.imshow(tmp, cmap=plt.cm.gray, alpha=0.5)
        plt.colorbar()
    plt.show()

    cv2.destroyAllWindows()
