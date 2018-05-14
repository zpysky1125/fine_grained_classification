from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from model import RecurrentAttentionModel
from tensorflow.examples.tutorials.mnist import input_data
import logging
import numpy as np

import sys

sys.path.append('../')

from input_generator import get_batch, BirdClassificationGenerator


train_image_num = 5094
valid_image_num = 900
test_image_num = 5794
train_batch_size = 8
valid_batch_size = 8
test_batch_size = 8

train_batch = train_image_num // train_batch_size if train_image_num % train_batch_size == 0 else train_image_num // train_batch_size + 1
valid_batch = valid_image_num // valid_batch_size if valid_image_num % valid_batch_size == 0 else valid_image_num // valid_batch_size + 1
test_batch = test_image_num // test_batch_size if test_image_num % test_batch_size == 0 else test_image_num // test_batch_size + 1


bird_classification_generator = BirdClassificationGenerator("../CUB_200_2011/CUB_200_2011/")
train_generator = bird_classification_generator.train_generator(train_batch_size)
valid_generator = bird_classification_generator.valid_generator(valid_batch_size)
test_generator = bird_classification_generator.test_generator(test_batch_size)

logging.getLogger().setLevel(logging.INFO)

# mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

tf.app.flags.DEFINE_float("learning_rate", 1e-3, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.97,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("min_learning_rate", 1e-4, "Minimum learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", train_batch_size, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("num_steps", 100000, "Number of training steps.")

tf.app.flags.DEFINE_integer("patch_window_size", 64, "Size of glimpse patch window.")
tf.app.flags.DEFINE_integer("g_size", 128, "Size of theta_g^0.")
tf.app.flags.DEFINE_integer("l_size", 128, "Size of theta_g^1.")
tf.app.flags.DEFINE_integer("glimpse_output_size", 1024, "Output size of Glimpse Network.")
tf.app.flags.DEFINE_integer("cell_size", 256, "Size of LSTM cell.")
tf.app.flags.DEFINE_integer("num_glimpses", 6, "Number of glimpses.")
tf.app.flags.DEFINE_float("variance", 0.22, "Gaussian variance for Location Network.")
tf.app.flags.DEFINE_integer("M", 10, "Monte Carlo sampling, see Eq(2).")

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



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in xrange(FLAGS.num_steps):

        images, labels = get_batch(train_generator, "../CUB_200_2011/CUB_200_2011/images/")
        images = np.tile(images, [FLAGS.M, 1, 1, 1])
        labels = np.tile(labels, [FLAGS.M])

        # images, labels = get_batch(train_generator)
        # images = np.tile(images, [FLAGS.M, 1])
        # labels = np.tile(labels, [FLAGS.M])

        output_feed = [ram.train_op, ram.loss, ram.xent, ram.reward, ram.advantage, ram.baselines_mse,
                       ram.learning_rate]
        _, loss, xent, reward, advantage, baselines_mse, learning_rate = sess.run(output_feed,
                                                                                  feed_dict={
                                                                                      ram.img_ph: images,
                                                                                      ram.lbl_ph: labels
                                                                                  })
        if step and step % 100 == 0:
            logging.info(
                'step {}: lr = {:3.6f}\tloss = {:3.4f}\txent = {:3.4f}\treward = {:3.4f}\tadvantage = {:3.4f}\tbaselines_mse = {:3.4f}'.format(
                    step, learning_rate, loss, xent, reward, advantage, baselines_mse))

        # Evaluation
        if step and step % train_batch == 0:
            for dataset in ['valid', 'test']:
                steps_per_epoch = valid_batch if dataset == 'valid' else test_batch
                correct_cnt = 0
                num_samples = steps_per_epoch * FLAGS.batch_size
                for test_step in xrange(steps_per_epoch):
                    images, labels = get_batch(valid_generator) if dataset == 'valid' else get_batch(test_generator)
                    labels_bak = labels
                    # Duplicate M times
                    images = np.tile(images, [FLAGS.M, 1])
                    labels = np.tile(labels, [FLAGS.M])
                    softmax = sess.run(ram.softmax,
                                       feed_dict={
                                           ram.img_ph: images,
                                           ram.lbl_ph: labels
                                       })
                    softmax = np.reshape(softmax, [FLAGS.M, -1, 10])
                    softmax = np.mean(softmax, 0)
                    prediction = np.argmax(softmax, 1).flatten()
                    correct_cnt += np.sum(prediction == labels_bak)
                acc = correct_cnt / num_samples
                if dataset == 'valid':
                    logging.info('valid accuracy = {}'.format(acc))
                else:
                    logging.info('test accuracy = {}'.format(acc))