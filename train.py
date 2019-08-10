import argparse
import logging
import os
import time

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from networks import get_network
from synthhands import SynthHands
from hands_dataset import get_dataflow_batch
from dataflow import DataFlowToQueue
from common import get_sample_images

logger = logging.getLogger('train')
logger.handlers.clear()
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training codes for Openpose using Tensorflow')
    parser.add_argument('--datapath', type=str, default='../hand_labels_synth')
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--input-width', type=int, default=368)
    parser.add_argument('--input-height', type=int, default=368)
    parser.add_argument('--gpus', type=int, default=4)
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--lr', type=str, default='0.0001')
    parser.add_argument('--max-epoch', type=int, default=600)
    parser.add_argument('--tag', type=str, default='test')
    args = parser.parse_args()

    modelpath = logpath = './models/train/'

    scale = 8
    output_w, output_h = args.input_width // scale, args.input_height // scale
    logger.info('define model+')
    with tf.device(tf.DeviceSpec(device_type="CPU")):
        input_node = tf.placeholder(tf.float32, shape=(args.batchsize, args.input_height, args.input_width, 3), name='image')
        heatmap_node = tf.placeholder(tf.float32, shape=(args.batchsize, output_h, output_w, 22), name='heatmap')

        # prepare data
        df = get_dataflow_batch(args.datapath, True, args.batchsize)
        enqueuer = DataFlowToQueue(df, [input_node, heatmap_node], queue_size=100)
        q_inp, q_heat  = enqueuer.dequeue()

    df_valid = get_dataflow_batch(args.datapath, False, args.batchsize)
    df_valid.reset_state()
    validation_cache = []

    val_image = get_sample_images(args.input_width, args.input_height)
    logger.debug('tensorboard val image: %d' % len(val_image))
    logger.debug(q_inp)
    logger.debug(q_heat)

    # define model for multi-gpu
    q_inp_split, q_heat_split  = tf.split(q_inp, args.gpus), tf.split(q_heat, args.gpus)

    output_vectmap = []
    output_heatmap = []
    losses = []
    last_losses_l2 = []
    outputs = []
    for gpu_id in range(args.gpus):
        with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_id)):
            with tf.variable_scope(tf.get_variable_scope(), reuse=(gpu_id > 0)):
                net, pretrain_path, last_layer = get_network("vgg", q_inp_split[gpu_id])
                if args.checkpoint:
                    pretrain_path = args.checkpoint
                heat = net.loss_last()
                output_heatmap.append(heat)
                outputs.append(net.get_output())

                l2s = net.loss_l2()
                for idx, l2 in enumerate(l2s):
                    loss_l2 = tf.nn.l2_loss(tf.concat(l2, axis=0) - q_heat_split[gpu_id], name='loss_l2_stage%d_tower%d' % (idx, gpu_id))
                    losses.append(tf.reduce_mean([loss_l2]))

                last_losses_l2.append(loss_l2)

    outputs = tf.concat(outputs, axis=0)

    with tf.device(tf.DeviceSpec(device_type="GPU")):
        # define loss
        total_loss = tf.reduce_sum(losses) / args.batchsize
        total_loss_ll_heat = tf.reduce_sum(last_losses_l2) / args.batchsize

        # define optimizer
        step_per_epoch = 121745 // args.batchsize
        global_step = tf.Variable(0, trainable=False)
        if ',' not in args.lr:
            starter_learning_rate = float(args.lr)
            # learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
            #                                            decay_steps=10000, decay_rate=0.33, staircase=True)
            learning_rate = tf.train.cosine_decay(starter_learning_rate, global_step, args.max_epoch * step_per_epoch, alpha=0.0)
        else:
            lrs = [float(x) for x in args.lr.split(',')]
            boundaries = [step_per_epoch * 5 * i for i, _ in range(len(lrs)) if i > 0]
            learning_rate = tf.train.piecewise_constant(global_step, boundaries, lrs)


    # optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=0.0005, momentum=0.9, epsilon=1e-10)
    optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=1e-8)
    # optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.8, use_locking=True, use_nesterov=True)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(total_loss, global_step, colocate_gradients_with_ops=True)
    logger.info('define model-')

    # define summary
    tf.summary.scalar("loss", total_loss)
    tf.summary.scalar("loss_lastlayer_heat", total_loss_ll_heat)
    tf.summary.scalar("queue_size", enqueuer.size())
    tf.summary.scalar("lr", learning_rate)
    merged_summary_op = tf.summary.merge_all()

    valid_loss = tf.placeholder(tf.float32, shape=[])
    valid_loss_ll_heat = tf.placeholder(tf.float32, shape=[])
    sample_train = tf.placeholder(tf.float32, shape=(4, 640, 640, 3))
    sample_valid = tf.placeholder(tf.float32, shape=(4, 640, 640, 3))
    train_img = tf.summary.image('training sample', sample_train, 4)
    valid_img = tf.summary.image('validation sample', sample_valid, 12)
    valid_loss_t = tf.summary.scalar("loss_valid", valid_loss)
    merged_validate_op = tf.summary.merge([train_img, valid_img, valid_loss_t])


    saver = tf.train.Saver(max_to_keep=1000)
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        logger.info('model weights initialization')
        sess.run(tf.global_variables_initializer())

        if args.checkpoint and os.path.isdir(args.checkpoint):
            logger.info('Restore from checkpoint...')
            # loader = tf.train.Saver(net.restorable_variables())
            # loader.restore(sess, tf.train.latest_checkpoint(args.checkpoint))
            saver.restore(sess, tf.train.latest_checkpoint(args.checkpoint))
            logger.info('Restore from checkpoint...Done')
        elif pretrain_path:
            logger.info('Restore pretrained weights... %s' % pretrain_path)
            if '.npy' in pretrain_path:
                net.load(pretrain_path, sess, True)
            else:
                try:
                    loader = tf.train.Saver(net.restorable_variables(only_backbone=False))
                    loader.restore(sess, pretrain_path)
                except:
                    logger.info('Restore only weights in backbone layers.')
                    loader = tf.train.Saver(net.restorable_variables())
                    loader.restore(sess, pretrain_path)
            logger.info('Restore pretrained weights...Done')

        logger.info('prepare file writer')
        file_writer = tf.summary.FileWriter(os.path.join(logpath, args.tag), sess.graph)

        logger.info('prepare coordinator')
        coord = tf.train.Coordinator()
        enqueuer.set_coordinator(coord)
        enqueuer.start()

        logger.info('Training Started.')
        time_started = time.time()
        last_gs_num = last_gs_num2 = 0
        initial_gs_num = sess.run(global_step)

        last_log_epoch1 = last_log_epoch2 = -1
        while True:
            _, gs_num = sess.run([train_op, global_step])
            curr_epoch = float(gs_num) / step_per_epoch

            if gs_num > step_per_epoch * args.max_epoch:
                break

            if gs_num - last_gs_num >= 500:
                train_loss, train_loss_ll_heat, lr_val, summary = sess.run([total_loss, total_loss_ll_heat, learning_rate, merged_summary_op])

                # log of training loss / accuracy
                batch_per_sec = (gs_num - initial_gs_num) / (time.time() - time_started)
                logger.info('epoch=%.2f step=%d, %0.4f examples/sec lr=%f, loss=%g, loss_ll_heat=%g' % (gs_num / step_per_epoch, gs_num, batch_per_sec * args.batchsize, lr_val, train_loss, train_loss_ll_heat))
                last_gs_num = gs_num

                if last_log_epoch1 < curr_epoch:
                    file_writer.add_summary(summary, curr_epoch)
                    last_log_epoch1 = curr_epoch

            if gs_num - last_gs_num2 >= 2000:
                # save weights
                saver.save(sess, os.path.join(modelpath, args.tag, 'model_latest'), global_step=global_step)

                average_loss = average_loss_ll_heat = 0
                total_cnt = 0

                if len(validation_cache) == 0:
                    for images_test, heatmaps in tqdm(df_valid.get_data()):
                        validation_cache.append((images_test, heatmaps))
                    df_valid.reset_state()
                    del df_valid
                    df_valid = None

                # log of test accuracy
                for images_test, heatmaps in validation_cache:
                    lss, lss_ll_heat, vectmap_sample, heatmap_sample = sess.run(
                        [total_loss, total_loss_ll_heat, output_vectmap, output_heatmap],
                        feed_dict={q_inp: images_test, q_heat: heatmaps}
                    )
                    average_loss += lss * len(images_test)
                    average_loss_ll_heat += lss_ll_heat * len(images_test)
                    total_cnt += len(images_test)

                logger.info('validation(%d) %s loss=%f, loss_ll_heat=%f' % (total_cnt, args.tag, average_loss / total_cnt, average_loss_ll_heat / total_cnt))
                last_gs_num2 = gs_num

                sample_image = [enqueuer.last_dp[0][i] for i in range(4)]
                outputMat = sess.run(
                    outputs,
                    feed_dict={q_inp: np.array((sample_image + val_image) * max(1, (args.batchsize // 8)))}
                )
                heatMat = outputMat[:, :, :, :19]

                sample_results = []
                for i in range(len(sample_image)):
                    test_result = SynthHands.display_image(sample_image[i], heatMat[i], as_numpy=True)
                    test_result = cv2.resize(test_result, (640, 640))
                    test_result = test_result.reshape([640, 640, 3]).astype(float)
                    sample_results.append(test_result)

                test_results = []
                for i in range(len(val_image)):
                    test_result = SynthHands.display_image(val_image[i], heatMat[len(sample_image) + i], as_numpy=True)
                    test_result = cv2.resize(test_result, (640, 640))
                    test_result = test_result.reshape([640, 640, 3]).astype(float)
                    test_results.append(test_result)

                # save summary
                summary = sess.run(merged_validate_op, feed_dict={
                    valid_loss: average_loss / total_cnt,
                    valid_loss_ll_heat: average_loss_ll_heat / total_cnt,
                    sample_valid: test_results,
                    sample_train: sample_results
                })
                if last_log_epoch2 < curr_epoch:
                    file_writer.add_summary(summary, curr_epoch)
                    last_log_epoch2 = curr_epoch

        saver.save(sess, os.path.join(modelpath, args.tag, 'model'), global_step=global_step)
    logger.info('optimization finished. %f' % (time.time() - time_started))
