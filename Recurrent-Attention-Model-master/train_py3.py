from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from model_py3 import RecurrentAttentionModel
from tensorflow.python import pywrap_tensorflow
import logging
import numpy as np

import sys

sys.path.append('../')

from input_generator import get_batch, BirdClassificationGenerator

train_image_num = 5094
valid_image_num = 900
test_image_num = 5794
train_batch_size = valid_batch_size = test_batch_size = 4

train_batch = train_image_num // train_batch_size if train_image_num % train_batch_size == 0 else train_image_num // train_batch_size + 1
valid_batch = valid_image_num // valid_batch_size if valid_image_num % valid_batch_size == 0 else valid_image_num // valid_batch_size + 1
test_batch = test_image_num // test_batch_size if test_image_num % test_batch_size == 0 else test_image_num // test_batch_size + 1


bird_classification_generator = BirdClassificationGenerator("../CUB_200_2011/CUB_200_2011/")
train_generator = bird_classification_generator.train_generator(train_batch_size)
valid_generator = bird_classification_generator.valid_generator(valid_batch_size)
test_generator = bird_classification_generator.test_generator(test_batch_size)

logging.getLogger().setLevel(logging.INFO)

# mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

tf.app.flags.DEFINE_float("learning_rate", 5e-4, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.97,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("min_learning_rate", 1e-5, "Minimum learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", train_batch_size, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("num_steps", 100000, "Number of training steps.")

tf.app.flags.DEFINE_integer("patch_window_size", 64, "Size of glimpse patch window.")
tf.app.flags.DEFINE_integer("g_size", 1024, "Size of theta_g^0.")
tf.app.flags.DEFINE_integer("l_size", 1024, "Size of theta_g^1.")
tf.app.flags.DEFINE_integer("glimpse_output_size", 4096, "Output size of Glimpse Network.")
tf.app.flags.DEFINE_integer("cell_size", 256, "Size of LSTM cell.")
tf.app.flags.DEFINE_integer("num_glimpses", 4, "Number of glimpses.")
tf.app.flags.DEFINE_float("variance", 0.22, "Gaussian variance for Location Network.")
tf.app.flags.DEFINE_integer("M", 5, "Monte Carlo sampling, see Eq(2).")

FLAGS = tf.app.flags.FLAGS

# training_steps_per_epoch = mnist.train.num_examples // FLAGS.batch_size

ram = RecurrentAttentionModel(img_size=224,  # MNIST: 28 * 28
                              pth_size=FLAGS.patch_window_size,
                              g_size=FLAGS.g_size,
                              l_size=FLAGS.l_size,
                              glimpse_output_size=FLAGS.glimpse_output_size,
                              loc_dim=2,  # (x,y)
                              variance=FLAGS.variance,
                              cell_size=FLAGS.cell_size,
                              num_glimpses=FLAGS.num_glimpses,
                              num_classes=200,
                              learning_rate=FLAGS.learning_rate,
                              learning_rate_decay_factor=FLAGS.learning_rate_decay_factor,
                              min_learning_rate=FLAGS.min_learning_rate,
                              training_steps_per_epoch=train_batch,
                              max_gradient_norm=FLAGS.max_gradient_norm,
                              is_training=True)


def get_checkpoint_tensor():
    a, b = 'GlimpseNetwork/conv', 'GlimpseNetwork/fc'
    var_list = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if not (a in v.name or b in v.name)]
    return var_list


saver1 = tf.train.Saver(var_list=get_checkpoint_tensor())
saver2 = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver1.restore(sess, 'tmp/model.ckpt-1596')

    for step in range(FLAGS.num_steps):
        images, labels = get_batch(train_generator, "../CUB_200_2011/CUB_200_2011/images/")
        images = np.tile(images, [FLAGS.M, 1, 1, 1])
        labels = np.tile(labels, [FLAGS.M])

        # images, labels = get_batch(train_generator)
        # images = np.tile(images, [FLAGS.M, 1])
        # labels = np.tile(labels, [FLAGS.M])

        # output_feed = [ram.train_op, ram.loss, ram.xent, ram.reward, ram.advantage, ram.baselines_mse,
        #                ram.learning_rate]

        # variables_names = [v.name for v in tf.trainable_variables()]
        # values = sess.run(variables_names)
        # for k, v in zip(variables_names, values):
        #     print ("Variable: ", k)
        #     print ("Shape: ", v.shape)
        #     print (v)

        output_feed = [ram.train_op, ram.loss, ram.xent, ram.reward, ram.advantage, ram.baselines_mse, ram.learning_rate]
        _, loss, xent, reward, advantage, baselines_mse, learning_rate = sess.run(output_feed,
                                                                                  feed_dict={
                                                                                      ram.img_ph: images,
                                                                                      ram.lbl_ph: labels,
                                                                                      ram.train_mode: True
                                                                                  })
        if step and step % 100 == 0:
            logging.info(
                'step {}: lr = {:3.6f}\tloss = {:3.4f}\txent = {:3.4f}\treward = {:3.4f}\tadvantage = {:3.4f}\tbaselines_mse = {:3.4f}'.format(
                    step, learning_rate, loss, xent, reward, advantage, baselines_mse))

        # Evaluation
        if step and step % train_batch == 0:
            for dataset in ['train', 'valid', 'test']:
                steps_per_epoch = None
                if dataset == 'valid':
                    steps_per_epoch = valid_batch
                elif dataset == 'test':
                    steps_per_epoch = test_batch
                else:
                    steps_per_epoch = train_batch
                # steps_per_epoch = valid_batch if dataset == 'valid' else
                correct_cnt = 0
                for test_step in range(steps_per_epoch):
                    images, labels = None, None
                    if dataset == 'valid':
                        images, labels = get_batch(valid_generator, "../CUB_200_2011/CUB_200_2011/images/")
                    elif dataset == 'test':
                        images, labels = get_batch(test_generator, "../CUB_200_2011/CUB_200_2011/images/")
                    else:
                        images, labels = get_batch(train_generator, "../CUB_200_2011/CUB_200_2011/images/")
                    # images, labels = get_batch(valid_generator, "../CUB_200_2011/CUB_200_2011/images/") if dataset == 'valid' else get_batch(test_generator, "../CUB_200_2011/CUB_200_2011/images/")
                    labels_bak = labels
                    # Duplicate M times
                    images = np.tile(images, [FLAGS.M, 1, 1, 1])
                    labels = np.tile(labels, [FLAGS.M])
                    softmax, logits, rnn_last, init_glip, glims, locs, rnn_out, rnn_state = sess.run([ram.softmax, ram.logits, ram.rnn_last, ram.init_glip, ram.glim, ram.locs, ram.rnn_out, ram.rnn_states],
                                       feed_dict={
                                           ram.img_ph: images,
                                           ram.lbl_ph: labels,
                                           ram.train_mode: False
                                       })
                    # print (init_glip)
                    # glim = np.transpose(glim, (1, 0, 2))
                    # for glim in glims:
                    #     print (glim)
                    # print (locs)
                    # print (rnn_last)
                    # rnn_out = np.transpose(rnn_out, (1, 0, 2))
                    # for out in rnn_out:
                    #     print (out)
                    # rnn_state = np.transpose(rnn_state, (1, 0, 2))
                    # for state in rnn_state:
                    #     print (state)
                    softmax = np.reshape(softmax, [FLAGS.M, -1, 200])
                    softmax = np.mean(softmax, 0)
                    prediction = np.argmax(softmax, 1).flatten()
                    # logging.info('prediction: {}'.format(prediction))
                    correct_cnt += np.sum(prediction == labels_bak)
                acc = None
                if dataset == 'valid':
                    acc = correct_cnt / valid_image_num
                elif dataset == 'test':
                    acc = correct_cnt / test_image_num
                else:
                    acc = correct_cnt / train_image_num
                # acc = correct_cnt / valid_image_num if dataset == 'valid' else correct_cnt / test_image_num
                if dataset == 'valid':
                    logging.info('valid accuracy = {}'.format(acc))
                elif dataset == 'test':
                    logging.info('test accuracy = {}'.format(acc))
                else:
                    logging.info('train accuracy = {}'.format(acc))
        if step and (step % (train_batch) == 0):
            save_path = saver2.save(sess, "tmp_fine_tune/model.ckpt", global_step=step+1)
