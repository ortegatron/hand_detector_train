from __future__ import absolute_import

import network_base
import tensorflow as tf

class CmuHandNetwork(network_base.BaseNetwork):
    def setup(self):
        (self.feed('image')
             .normalize_vgg(name='preprocess')
             .conv(3, 3, 64, 1, 1, name='conv1_1')
             .conv(3, 3, 64, 1, 1, name='conv1_2')
             .max_pool(2, 2, 2, 2, name='pool1_stage1', padding='VALID')
             .conv(3, 3, 128, 1, 1, name='conv2_1')
             .conv(3, 3, 128, 1, 1, name='conv2_2')
             .max_pool(2, 2, 2, 2, name='pool2_stage1', padding='VALID')
             .conv(3, 3, 256, 1, 1, name='conv3_1')
             .conv(3, 3, 256, 1, 1, name='conv3_2')
             .conv(3, 3, 256, 1, 1, name='conv3_3')
             .conv(3, 3, 256, 1, 1, name='conv3_4')
             .max_pool(2, 2, 2, 2, name='pool3_stage1', padding='VALID')
             .conv(3, 3, 512, 1, 1, name='conv4_1')
             .conv(3, 3, 512, 1, 1, name='conv4_2')
             .conv(3, 3, 512, 1, 1, name='conv4_3')
             .conv(3, 3, 512, 1, 1, name='conv4_4')
             .conv(3, 3, 512, 1, 1, name='conv5_1')
             .conv(3, 3, 512, 1, 1, name='conv5_2')
             .conv(3, 3, 128, 1, 1, name='conv5_3_CPM')          # *****
             .conv(3, 3, 512, 1, 1, name='conv6_1_CPM')
             .conv(3, 3, 22, 1, 1, relu = False, name='conv6_2_CPM'))

        (self.feed('conv5_3_CPM',
                   'conv6_2_CPM',)
             .concat(3, name='concat_stage2')
             .conv(7, 7, 128, 1, 1, name='Mconv1_stage2')
             .conv(7, 7, 128, 1, 1, name='Mconv2_stage2')
             .conv(7, 7, 128, 1, 1, name='Mconv3_stage2')
             .conv(7, 7, 128, 1, 1, name='Mconv4_stage2')
             .conv(7, 7, 128, 1, 1, name='Mconv5_stage2')
             .conv(1, 1, 128, 1, 1, name='Mconv6_stage2')
             .conv(1, 1, 22, 1, 1, relu=False, name='Mconv7_stage2'))

        (self.feed('conv5_3_CPM',
                   'Mconv7_stage2',)
             .concat(3, name='concat_stage3')
             .conv(7, 7, 128, 1, 1, name='Mconv1_stage3')
             .conv(7, 7, 128, 1, 1, name='Mconv2_stage3')
             .conv(7, 7, 128, 1, 1, name='Mconv3_stage3')
             .conv(7, 7, 128, 1, 1, name='Mconv4_stage3')
             .conv(7, 7, 128, 1, 1, name='Mconv5_stage3')
             .conv(1, 1, 128, 1, 1, name='Mconv6_stage3')
             .conv(1, 1, 22, 1, 1, relu=False, name='Mconv7_stage3'))

        (self.feed('conv5_3_CPM',
                   'Mconv7_stage3',)
             .concat(3, name='concat_stage4')
             .conv(7, 7, 128, 1, 1, name='Mconv1_stage4')
             .conv(7, 7, 128, 1, 1, name='Mconv2_stage4')
             .conv(7, 7, 128, 1, 1, name='Mconv3_stage4')
             .conv(7, 7, 128, 1, 1, name='Mconv4_stage4')
             .conv(7, 7, 128, 1, 1, name='Mconv5_stage4')
             .conv(1, 1, 128, 1, 1, name='Mconv6_stage4')
             .conv(1, 1, 22, 1, 1, relu=False, name='Mconv7_stage4'))

        (self.feed('conv5_3_CPM',
                   'Mconv7_stage4',)
             .concat(3, name='concat_stage5')
             .conv(7, 7, 128, 1, 1, name='Mconv1_stage5')
             .conv(7, 7, 128, 1, 1, name='Mconv2_stage5')
             .conv(7, 7, 128, 1, 1, name='Mconv3_stage5')
             .conv(7, 7, 128, 1, 1, name='Mconv4_stage5')
             .conv(7, 7, 128, 1, 1, name='Mconv5_stage5')
             .conv(1, 1, 128, 1, 1, name='Mconv6_stage5')
             .conv(1, 1, 22, 1, 1, relu=False, name='Mconv7_stage5'))

        (self.feed('conv5_3_CPM',
                   'Mconv7_stage5',)
             .concat(3, name='concat_stage6')
             .conv(7, 7, 128, 1, 1, name='Mconv1_stage6')
             .conv(7, 7, 128, 1, 1, name='Mconv2_stage6')
             .conv(7, 7, 128, 1, 1, name='Mconv3_stage6')
             .conv(7, 7, 128, 1, 1, name='Mconv4_stage6')
             .conv(7, 7, 128, 1, 1, name='Mconv5_stage6')
             .conv(1, 1, 128, 1, 1, name='Mconv6_stage6'))

        (self.feed('Mconv6_stage6')
            .conv(1, 1, 22, 1, 1, relu=False, name='Mconv7_stage6'))

        with tf.variable_scope('Openpose'):
            (self.feed('Mconv7_stage6')
                 .concat(3, name='out'))


    def loss_l2(self):
         l2s = []
         for layer_name in self.layers.keys():
              if 'Mconv7' in layer_name:
                   l2s.append(self.layers[layer_name])
         return l2s

    def loss_last(self):
         return self.get_output('Mconv7_stage6')

    def restorable_variables(self):
         return None
