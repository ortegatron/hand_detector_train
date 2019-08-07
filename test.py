import tensorflow as tf
import sys
import cv2
import os
from sys import platform
import argparse
import matplotlib.pyplot as plt
import numpy as np
from networks import get_network



def read_imgfile(path, width, height):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    # val_img = preprocess(img, width, height)

    return img.astype(np.float16)

def preprocess(img, width, height):
    val_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # cv2 reads in BGR format
    val_img = cv2.resize(val_img, (width, height)) # each net accept only a certain size
    val_img = val_img.reshape([1, height, width, 3])
    val_img = val_img.astype(float)
    val_img = val_img * (2.0 / 255.0) - 1.0 # image range from -1 to +1
    return val_img

### IMAGE ###
# img = read_imgfile('/home/marcelo/hands/hand_labels_synth/synth1/2183.jpg',368,368)
file_img = "./real.jpeg"
file_img="../hand_labels_synth/synth2/00000005.jpg"
file_img="../hand_labels_synth/synth2/00000131.jpg"
img = [read_imgfile(file_img,368,368)]
img_raw = cv2.imread(file_img)

input_node = tf.placeholder(tf.float32, shape=(1, 368, 368, 3), name='image')
net, pretrain_path, _ = get_network("vgg", input_node)


outputs = []
outputs.append(net.get_output())
outputs = tf.concat(outputs, axis=0)

saver = tf.train.Saver(max_to_keep=1000)
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    print(tf.train.latest_checkpoint('./models/train/test/'))
    saver.restore(sess, tf.train.latest_checkpoint('./models/train/test/'))
    outputMat = sess.run(
        outputs,
        feed_dict={input_node: np.array(img)}
    )
    heatMat = outputMat[:, :, :, :19][0]




# print(type(output_img))
# print(output_img.shape)
# fig, ax = plt.subplots(nrows=5, ncols=5)
# index = 0
# for row in ax:
#     for col in row:
#         col.imshow(output_img[0,:,:,index])
#         index += 1
#         if index >= 22:
#             break
# plt.show()


from synthhands import SynthHands
print(img.shape)
print(heatMat.shape)
test_result = SynthHands.display_image(img[0], heatMat, as_numpy=True)
test_result = cv2.cvtColor(test_result, cv2.COLOR_RGB2BGR)
cv2.imshow("hi",test_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
