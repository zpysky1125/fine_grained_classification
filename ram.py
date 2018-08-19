from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import logging
import os
import time
import sys
import config
from tensorflow.python.ops.rnn_cell_impl import BasicLSTMCell
from tensorflow.python.ops.distributions.normal import Normal
from tensorflow.contrib import slim
from tensorflow.contrib.slim.nets import resnet_v1
from PIL import Image


class RecurrentAttentionModel(object):
    def __init__(self, sess=None, batch_size=64, multi_batch_size=16, pth_size=64, glimpse_output_size=1024,
                 cell_size=256, variance=0.22, num_glimpses=5, glimpse_times=5, drop1=0.3, drop2=0.3, train_method='Adam',
                 crop_or_mask='crop', result_mode='single', reinforce_mode='baseline', learning_rate=5e-4,
                 learning_rate_decay_factor=0.97, min_learning_rate=1e-5, max_gradient_norm=5.0, resize_side_min=224,
                 resize_side_max=512, train_path='', valid_path='', test_path='', logging_step=50, train_epoch=20,
                 train_step_per_epoch=200, test_step_per_epoch=100, valid_step_per_epoch=100, test_multi_step_per_epoch=25):

        self.batch_size = batch_size
        self.multi_batch_size = multi_batch_size

        self.train_step_per_epoch = (5094 // self.batch_size) if 5094 % self.batch_size == 0 else (5094 // self.batch_size + 1)
        self.test_step_per_epoch = (5794 // self.batch_size) if 5794 % self.batch_size == 0 else (5794 // self.batch_size + 1)
        self.test_multi_step_per_epoch = (5794 // self.multi_batch_size) if 5794 % self.multi_batch_size == 0 else (5794 // self.multi_batch_size + 1)
        self.valid_step_per_epoch = (900 // self.batch_size) if 900 % self.batch_size == 0 else (900 // self.batch_size + 1)

        self.img_size = 224
        self.pth_size = pth_size

        self.lstm_cell_size = cell_size
        self.glimpse_output_size = glimpse_output_size

        self.loc_dim = 2

        self.variance = variance

        self.num_glimpses = num_glimpses
        self.glimpse_times = glimpse_times

        self.drop1 = drop1
        self.drop2 = drop2

        self.train_method = train_method
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.learning_rate = learning_rate
        self.decay_learning_rate = tf.maximum(tf.train.exponential_decay(
            learning_rate, self.global_step,
            self.train_step_per_epoch,
            learning_rate_decay_factor,
            staircase=True),
            min_learning_rate)
        self.max_gradient_norm = max_gradient_norm

        self.resize_side_min = resize_side_min
        self.resize_side_max = resize_side_max

        self.train_path = train_path
        self.valid_path = valid_path
        self.test_path = test_path

        self.train_image_names, self.train_labels = self._generate_image_names_and_labels(train_path)
        self.test_image_names, self.test_labels = self._generate_image_names_and_labels(test_path)
        self.valid_image_names, self.valid_labels = self._generate_image_names_and_labels(valid_path)

        self.train_epoch = train_epoch

        self.logging_step = logging_step

        self.mode = sys.argv[1]

        self.crop_or_mask = crop_or_mask
        self.result_mode = result_mode
        self.pth_mode = 'fix'
        self.reinforce_mode = reinforce_mode

        self.sess = sess

        self.model()

    def _generate_image_names_and_labels(self, train_path):
        image_names, labels = [], []
        with open(train_path) as f:
            for line in f.readlines():
                image_names.append(os.path.join('CUB_200_2011/CUB_200_2011/images', line.strip()))
                labels.append(int(line.strip().split('.')[0]) - 1)
        return image_names, labels

    def _smallest_size_at_least(self, height, width, smallest_side):
        smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

        height = tf.to_float(height)
        width = tf.to_float(width)
        smallest_side = tf.to_float(smallest_side)

        scale = tf.cond(tf.greater(height, width),
                        lambda: smallest_side / width,
                        lambda: smallest_side / height)
        new_height = tf.to_int32(tf.rint(height * scale))
        new_width = tf.to_int32(tf.rint(width * scale))
        return new_height, new_width

    def _aspect_preserving_resize(self, image, smallest_side):
        smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)
        shape = tf.shape(image)
        height, width = shape[0], shape[1]
        new_height, new_width = self._smallest_size_at_least(height, width, smallest_side)
        image = tf.expand_dims(image, 0)
        resized_image = tf.image.resize_bilinear(image, [new_height, new_width], align_corners=False)
        resized_image = tf.squeeze(resized_image)
        resized_image.set_shape([None, None, 3])
        return resized_image

    def _mean_image_subtraction(self, image, means):
        num_channels = image.get_shape().as_list()[-1]
        channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
        for i in range(num_channels):
            channels[i] -= means[i]
        return tf.concat(axis=2, values=channels)

    def _train_parse_func(self, filename, label):
        image_string = tf.read_file(filename)
        image = tf.image.decode_jpeg(image_string, channels=3)
        if self.mode == 'origin':
            image = tf.image.resize_images(image, [448, 448])
        else:
            resize_side = tf.random_uniform([], minval=self.resize_side_min, maxval=self.resize_side_max+1, dtype=tf.int32)
            image = self._aspect_preserving_resize(image, resize_side)
            crop = tf.random_crop(image, [224, 224, 3])
            image = tf.to_float(crop)
        normalize_image = self._mean_image_subtraction(image, [123.68, 116.78, 103.94])
        normalize_image = tf.image.random_flip_left_right(normalize_image)
        # return image, label
        return normalize_image, label

    def _test_parse_func(self, filename, label):
        image_string = tf.read_file(filename)
        image = tf.image.decode_jpeg(image_string, channels=3)
        if self.mode == 'origin':
            image = tf.image.resize_images(image, [448, 448])
        else:
            image = self._aspect_preserving_resize(image, self.resize_side_min)
            crop = tf.image.resize_image_with_crop_or_pad(image, 224, 224)
            image = tf.to_float(crop)
        normalize_image = self._mean_image_subtraction(image, [123.68, 116.78, 103.94])
        return normalize_image, label

    def _test_multi_parse_func(self, filename, label):
        image_string = tf.read_file(filename)
        image = tf.image.decode_jpeg(image_string, channels=3)
        image = self._aspect_preserving_resize(image, self.resize_side_min)
        center_crop = tf.image.resize_image_with_crop_or_pad(image, 224, 224)
        # multi crop
        height, width = tf.shape(image)[0], tf.shape(image)[1]
        top_left_crop = tf.image.crop_to_bounding_box(image, 0, 0, 224, 224)
        top_right_crop = tf.image.crop_to_bounding_box(image, 0, width-224, 224, 224)
        bottom_left_crop = tf.image.crop_to_bounding_box(image, height-224, 0, 224, 224)
        bottom_right_crop = tf.image.crop_to_bounding_box(image, height-224, width-224, 224, 224)
        multi_crops = [center_crop, top_left_crop, top_right_crop, bottom_left_crop, bottom_right_crop]
        multi_crops = [self._mean_image_subtraction(tf.to_float(crop), [123.68, 116.78, 103.94]) for crop in multi_crops]
        flip_crops = [tf.image.flip_left_right(crop) for crop in multi_crops]
        multi_crops = multi_crops + flip_crops
        multi_crops = tf.stack(multi_crops)
        return multi_crops, label

    def _generate_train_image_label(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.train_image_names, self.train_labels))
        dataset = dataset.map(self._train_parse_func)
        dataset = dataset.shuffle(buffer_size=500)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.repeat()
        return dataset

    def generate_test_image_label(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.test_image_names, self.test_labels))
        dataset = dataset.map(self._test_parse_func)
        dataset = dataset.batch(self.batch_size)
        return dataset

    def generate_valid_image_label(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.valid_image_names, self.valid_labels))
        dataset = dataset.map(self._test_parse_func)
        dataset = dataset.batch(self.batch_size)
        return dataset

    def generate_test_multi_image_label(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.test_image_names, self.test_labels))
        dataset = dataset.map(self._test_multi_parse_func)
        dataset = dataset.batch(self.multi_batch_size)
        return dataset

    def glimpse_network(self, img, loc, init):
        def retina_sensor(img_ph, loc, init=False):
            ll = tf.stop_gradient(loc)
            if init:
                pth = img_ph
            else:
                pth = tf.image.extract_glimpse(img_ph, [self.pth_size, self.pth_size], ll)
                if self.crop_or_mask == 'crop':
                    pth = tf.image.resize_images(pth, [224, 224])
                else:
                    pth = tf.image.pad_to_bounding_box(pth, int((224-self.pth_size)/2), int((224-self.pth_size)/2), 224, 224)
            return pth

        def feature_extractor(patch):
            with slim.arg_scope(resnet_v1.resnet_arg_scope()):
                net, end_points = resnet_v1.resnet_v1_50(patch, 1000, is_training=self.train_mode, reuse=tf.AUTO_REUSE)
            resnet_feature = end_points['resnet_v1_50/block4']
            resnet_feature = tf.reduce_mean(resnet_feature, [1, 2], keepdims=True)
            resnet_feature = tf.squeeze(resnet_feature)
            resnet_feature = tf.reshape(resnet_feature, [-1, 2048])
            drop = tf.layers.dropout(resnet_feature, rate=self.drop1, training=self.train_mode)
            glimpse_feature = tf.layers.dense(inputs=drop, units=512, activation=tf.nn.relu,
                                              kernel_initializer=tf.glorot_uniform_initializer(),
                                              bias_initializer=tf.constant_initializer(0.1),
                                              # kernel_regularizer=tf.nn.l2_loss,
                                              name='glimpse_feature/fc', reuse=tf.AUTO_REUSE)
            return glimpse_feature

        def loc_feature_extractor(locs):
            with tf.name_scope("loc_feature_extractor"):
                location_feature = tf.layers.dense(locs, self.glimpse_output_size,
                                                   kernel_initializer=tf.glorot_uniform_initializer(),
                                                   bias_initializer=tf.constant_initializer(0.1),
                                                   # kernel_regularizer=tf.nn.l2_loss,
                                                   name='location_feature/fc', reuse=tf.AUTO_REUSE)
            return location_feature

        rgb = retina_sensor(img, loc, init)
        g = feature_extractor(rgb)
        return rgb, tf.nn.relu(g)
        # l = loc_feature_extractor(loc)
        # return rgb, tf.nn.relu(tf.concat([l, g], axis=-1))
        # return rgb, tf.nn.relu(l+g)

    def location_network(self, cell_output, is_sampling):
        mean = tf.layers.dense(cell_output, self.loc_dim, kernel_initializer=tf.glorot_uniform_initializer(),
                               bias_initializer=tf.constant_initializer(0.1),
                               # kernel_regularizer=tf.nn.l2_loss,
                               name='location/fc', reuse=tf.AUTO_REUSE)
        mean = tf.clip_by_value(mean, -1., 1.)

        if is_sampling:
            loc = mean + tf.random_normal((tf.shape(cell_output)[0], self.loc_dim), stddev=self.variance)
            loc = tf.clip_by_value(loc, -1., 1.)
        else:
            loc = mean
        return loc, mean

    def baseline_network(self, rnn_outputs):
        baseline = tf.layers.dense(tf.stack(rnn_outputs), 1, kernel_initializer=tf.glorot_uniform_initializer(),
                                   bias_initializer=tf.constant_initializer(0.1),
                                   # kernel_regularizer=tf.nn.l2_loss,
                                   name='baseline/fc', reuse=tf.AUTO_REUSE)
        if self.num_glimpses != 1:
            baseline = tf.transpose(tf.squeeze(baseline), (1, 0))
        return baseline

    def prob_network(self, rnn_last_out):
        last_out = tf.layers.dropout(rnn_last_out, rate=self.drop2, training=self.train_mode)
        logits = tf.layers.dense(last_out, 200, kernel_initializer=tf.glorot_uniform_initializer(),
                                 bias_initializer=tf.constant_initializer(0.1),
                                 # kernel_regularizer=tf.nn.l2_loss,
                                 name='prob/fc', reuse=tf.AUTO_REUSE)
        probs = tf.nn.softmax(logits)
        return logits, probs

    def loop_function(self, prev, step, img_ph):
        loc, loc_mean = self.location_network(prev, True)
        if step > 0:
            rgb, glimpse = self.glimpse_network(img_ph, loc, False)
        else:
            rgb, glimpse = self.glimpse_network(img_ph, loc, True)
        return rgb, glimpse, loc, loc_mean

    def rnn_decode(self, initial_state, cell, img_ph):
        state = initial_state
        outputs, locs, loc_means = [], [], []
        rgbs = []
        prev = tf.zeros([tf.shape(img_ph)[0], self.lstm_cell_size])
        for i in range(self.num_glimpses):
            rgb, inp, loc, loc_mean = self.loop_function(prev, i, img_ph)
            output, state = cell(inp, state)
            prev = output
            outputs.append(output)
            locs.append(loc)
            loc_means.append(loc_mean)
            rgbs.append(rgb)
        return rgbs, outputs, locs, loc_means

    def _log_likelihood(self, loc_means, locs, variance):
        loc_means = tf.stack(loc_means)
        locs = tf.stack(locs)
        gaussian = Normal(loc_means, variance)
        logll = gaussian._log_prob(x=locs)
        logll = tf.reduce_sum(logll, 2)
        return tf.transpose(logll)

    def _loc_loss(self, loc_means):
        loss = 0.0
        for i in range(self.num_glimpses):
            for j in range(i, self.num_glimpses):
                loss += tf.reduce_mean(tf.square(loc_means[i] - loc_means[j]))
        return loss

    def model(self):
        self.train_mode = tf.placeholder(tf.bool)
        self.train_dataset = self._generate_train_image_label()
        self.test_dataset = self.generate_test_image_label()
        self.valid_dataset = self.generate_valid_image_label()

        self.iter = tf.data.Iterator.from_structure(self.train_dataset.output_types, self.train_dataset.output_shapes)
        self.train_init_op = self.iter.make_initializer(self.train_dataset)
        self.test_init_op = self.iter.make_initializer(self.test_dataset)
        self.valid_init_op = self.iter.make_initializer(self.valid_dataset)

        self.images, self.labels = self.iter.get_next()

        self.img_ph = tf.tile(self.images, [self.glimpse_times, 1, 1, 1])
        self.lbl_ph = tf.tile(self.labels, [self.glimpse_times])

        cell = BasicLSTMCell(self.lstm_cell_size)
        init_state = cell.zero_state(tf.shape(self.img_ph)[0], tf.float32)
        self.rgbs, self.rnn_outputs, self.locs, self.loc_means = self.rnn_decode(init_state, cell, self.img_ph)

        baselines = self.baseline_network(self.rnn_outputs)
        logits, probs = self.prob_network(self.rnn_outputs[-1])
        predict_label = tf.argmax(logits, 1, output_type=tf.int32)

        if self.result_mode == 'single':
            rewards = tf.cast(tf.equal(predict_label, self.lbl_ph), tf.float32)
            rewards = tf.tile(tf.expand_dims(rewards, 1), [1, self.num_glimpses])
            self.test_label = tf.argmax(tf.reduce_mean(tf.reshape(probs, [self.glimpse_times, -1, 200]), axis=0), 1,
                                        output_type=tf.int32)
            self.test_acc = tf.reduce_mean(tf.cast(tf.equal(self.test_label, self.labels), tf.float32))
            self.m_test_acc = self.test_acc
        else:
            m_logits, m_probs = self.prob_network(tf.stack(self.rnn_outputs))
            m_predict_label = tf.argmax(m_logits, -1, output_type=tf.int32)
            m_lbl = tf.tile(tf.expand_dims(self.lbl_ph, 0), [self.num_glimpses, 1])
            rewards = tf.transpose(tf.cast(tf.equal(m_predict_label, m_lbl), tf.float32), [1, 0])
            self.classification_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=m_lbl, logits=m_logits))
            self.test_label = tf.argmax(tf.reduce_mean(tf.reshape(probs, [self.glimpse_times, -1, 200]), axis=0), 1,
                                        output_type=tf.int32)
            self.test_acc = tf.reduce_mean(tf.cast(tf.equal(self.test_label, self.labels), tf.float32))
            m_test_label = tf.argmax(tf.reduce_mean(tf.reshape(tf.reduce_mean(m_probs, 0),
                                        [self.glimpse_times, -1, 200]), axis=0), 1, output_type=tf.int32)
            self.m_test_acc = tf.reduce_mean(tf.cast(tf.equal(m_test_label, self.labels), tf.float32))

        log_action_prob = self._log_likelihood(self.loc_means, self.locs, self.variance)

        if self.reinforce_mode == 'baseline':
            advantages = tf.stop_gradient(rewards - baselines)
            if self.result_mode == 'single':
                self.classification_loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.lbl_ph, logits=logits))
            self.reinforce_loss = tf.reduce_mean(log_action_prob * advantages)
            self.baselines_loss = tf.reduce_mean(tf.square((tf.stop_gradient(rewards) - baselines)))
            self.location_loss = self._loc_loss(self.loc_means)
            self.loss = -self.reinforce_loss + self.classification_loss + self.baselines_loss

        else:
            advantages = rewards
            if self.result_mode == 'single':
                self.classification_loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.lbl_ph, logits=logits))
            self.reinforce_loss = tf.reduce_mean(log_action_prob * advantages)
            self.baselines_loss = tf.constant(0)
            self.location_loss = self._loc_loss(self.loc_means)
            self.loss = -self.reinforce_loss + self.classification_loss

        if self.mode == 'fine_tune':
            params = tf.trainable_variables()
        elif self.mode == 'origin':
            params = [param for param in tf.trainable_variables() if 'resnet' not in param.name]
        else:
            # params = tf.trainable_variables()
            params = [param for param in tf.trainable_variables() if 'resnet' not in param.name]
            # params = [param for param in tf.trainable_variables() if 'resnet' not in param.name and
            #           'glimpse_feature' not in param.name]
        gradients = tf.gradients(self.loss, params)
        clipped_gradients, norm = tf.clip_by_global_norm(gradients, self.max_gradient_norm)

        if self.train_method == 'Adam':
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.decay_learning_rate).\
                apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)
        else:
            self.train_op = tf.train.MomentumOptimizer(self.decay_learning_rate, 0.9). \
                apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)


        self.reward = tf.reduce_mean(rewards)
        self.advantage = tf.reduce_mean(advantages)

        tf.summary.histogram('probs', tf.reduce_sum(probs * tf.one_hot(self.lbl_ph, 200), axis=-1))
        for param in params:
            tf.summary.histogram(param.name, param)
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('reward', self.reward)
        tf.summary.scalar('advantage', self.advantage)
        tf.summary.scalar('classification_loss', self.classification_loss)
        tf.summary.scalar('reinforce_loss', self.reinforce_loss)
        tf.summary.scalar('baseline_loss', self.baselines_loss)
        self.merged = tf.summary.merge_all()

        self.test_multi_dataset = self.generate_test_multi_image_label()
        self.multi_iter = tf.data.Iterator.from_structure(self.test_multi_dataset.output_types, self.test_multi_dataset.output_shapes)
        self.test_multi_init_op = self.multi_iter.make_initializer(self.test_multi_dataset)

        self.multi_images, self.multi_labels = self.multi_iter.get_next()
        self.multi_images = tf.reshape(self.multi_images, [-1, 224, 224, 3])
        self.multi_img_ph = tf.tile(self.multi_images, [self.glimpse_times, 1, 1, 1])
        multi_init_state = cell.zero_state(tf.shape(self.multi_img_ph)[0], tf.float32)
        _, self.multi_rnn_outputs, __, ___ = self.rnn_decode(multi_init_state, cell, self.multi_img_ph)
        _, multi_probs = self.prob_network(self.multi_rnn_outputs[-1])
        self.multi_test_labels = tf.reduce_mean(tf.reshape(multi_probs, [self.glimpse_times, -1, 200]), axis=0)
        self.multi_test_labels = tf.reduce_mean(tf.reshape(self.multi_test_labels, [-1, 10, 200]), axis=1)
        self.multi_test_labels = tf.argmax(self.multi_test_labels, 1, output_type=tf.int32)
        self.multi_accuracy = tf.reduce_mean(tf.cast(tf.equal(self.multi_test_labels, self.multi_labels), tf.float32))

    def run(self):
        self.sess.run(tf.global_variables_initializer())
        if self.mode == 'init':
            var_list = [var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if ('resnet' in var.name
                        or 'glimpse_feature' in var.name) and self.train_method not in var.name]
            restore = tf.train.Saver(var_list)
            restore.restore(self.sess, sys.argv[2])
        elif self.mode == 'origin':
            init_fn = slim.assign_from_checkpoint_fn('resnet_v1_50.ckpt', slim.get_model_variables('resnet_v1_50'))
            init_fn(self.sess)
        elif self.mode == 'fine_tune':
            var_list = [var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if self.train_method not in var.name
                        and 'global_step' not in var.name]
            restore = tf.train.Saver(var_list)
            restore.restore(self.sess, sys.argv[2])
        else:
            sys.exit(1)

        saver = tf.train.Saver(max_to_keep=30)
        cur_time = str(time.time())
        train_writer = tf.summary.FileWriter('log/ram/' + self.mode + '_' + str(self.learning_rate) + '_' + cur_time + '/board/train', self.sess.graph)
        test_writer = tf.summary.FileWriter('log/ram/' + self.mode + '_' + str(self.learning_rate) + '_' + cur_time + '/board/test', self.sess.graph)
        valid_writer = tf.summary.FileWriter('log/ram/' + self.mode + '_' + str(self.learning_rate) + '_' + cur_time + '/board/valid', self.sess.graph)

        logging.info(self.mode)
        logging.info(str(self.learning_rate))
        logging.info(cur_time)
        logging.info(self.train_method)
        logging.info('{} {}'.format(self.num_glimpses, self.glimpse_times))
        logging.info('{} {} {} {}'.format(self.crop_or_mask, self.result_mode, self.pth_mode, self.reinforce_mode))
        logging.info('{} {} {} {} {} {} {}'.format(self.drop1, self.drop2, self.pth_size, self.lstm_cell_size, self.glimpse_output_size,
                                                self.resize_side_min, self.resize_side_max))

        for epoch in range(self.train_epoch):
            statistics = []
            self.sess.run(self.train_init_op)
            for train_step in range(self.train_step_per_epoch):
                _, summary, loss, reward, advantage, cls_loss, rein_loss, base_loss = self.sess.run(
                    [self.train_op, self.merged, self.loss, self.reward, self.advantage, self.classification_loss,
                     self.reinforce_loss, self.baselines_loss],
                    feed_dict={self.train_mode: True})

                train_writer.add_summary(summary, epoch * self.train_step_per_epoch + train_step)
                statistics.append([loss, reward, advantage, cls_loss, rein_loss, base_loss])

                if train_step and train_step % self.logging_step == 0:
                    statistics = np.mean(np.asarray(statistics, np.float32), axis=0)
                    logging.info('Epoch {} Train step {}: loss = {:3.4f}\t reward = {:3.4f} \t advantage = {:3.4f} \t '
                                 'cls_loss = {:3.4f} \t rein_loss = {:3.4f} \t base_loss = {:3.4f}'
                                 .format(epoch, train_step, *statistics))
                    statistics = []

            self.sess.run(self.valid_init_op)
            statistics = []
            for valid_step in range(self.valid_step_per_epoch):
                summary, loss, acc, m_test_acc, reward, advantage, cls_loss, rein_loss, base_loss = self.sess.run(
                    [self.merged, self.loss, self.test_acc, self.m_test_acc, self.reward, self.advantage,
                     self.classification_loss, self.reinforce_loss, self.baselines_loss],
                    feed_dict={self.train_mode: False})
                valid_writer.add_summary(summary, epoch * self.valid_step_per_epoch + valid_step)
                statistics.append([loss, acc, m_test_acc, reward, advantage, cls_loss, rein_loss, base_loss])
            statistics = np.mean(np.asarray(statistics, np.float32), axis=0)
            logging.info('Epoch {} Valid: loss = {:3.4f}\t acc = {:3.4f} \t m_acc = {:3.4f}\t reward = {:3.4f} \t '
                         'advantage = {:3.4f} \t cls_loss = {:3.4f} \t rein_loss = {:3.4f} \t base_loss = {:3.4f}'
                         .format(epoch, *statistics))

            if epoch and epoch % 5 == 0:
                self.sess.run(self.test_init_op)
                statistics = []
                for test_step in range(self.test_step_per_epoch):
                    summary, loss, acc, m_test_acc, reward, advantage, cls_loss, rein_loss, base_loss, loc_means = self.sess.run(
                        [self.merged, self.loss, self.test_acc, self.m_test_acc, self.reward, self.advantage, self.classification_loss,
                         self.reinforce_loss, self.baselines_loss, self.loc_means],
                        feed_dict={self.train_mode: False})
                    test_writer.add_summary(summary, (epoch / 5) * self.test_step_per_epoch + test_step)
                    statistics.append([loss, acc, m_test_acc, reward, advantage, cls_loss, rein_loss, base_loss])
                statistics = np.mean(np.asarray(statistics, np.float32), axis=0)
                logging.info('Epoch {} Test: loss = {:3.4f}\t acc = {:3.4f} \t m_acc = {:3.4f} \t reward = {:3.4f} \t '
                             'advantage = {:3.4f} \t cls_loss = {:3.4f} \t rein_loss = {:3.4f} \t base_loss = {:3.4f}'
                             .format(epoch, *statistics))

                # if epoch % 20 == 0:
                #     self.sess.run(self.test_multi_init_op)
                #     multi_accs = []
                #     for test_step in range(self.test_multi_step_per_epoch):
                #         multi_acc = self.sess.run(self.multi_accuracy, feed_dict={self.train_mode: False})
                #         multi_accs.append(multi_acc)
                #     logging.info('Epoch {} Test Multi: accuracy = {:3.4f}'.format(epoch, np.mean(multi_accs)))

                saver.save(self.sess, 'log/ram/' + self.mode + '_' + str(self.learning_rate) + '_' + cur_time +
                           '/tmp/model.ckpt', global_step=epoch)

        train_writer.close()
        test_writer.close()
        valid_writer.close()


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    with tf.Session() as sess:
        model = RecurrentAttentionModel(sess, **config.recurrent_attention_model)
        model.run()
