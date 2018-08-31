from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import time
import sys
import config
from tensorflow.contrib.distributions import Normal
from tensorflow.contrib import slim
from tensorflow.contrib.slim.nets import resnet_v1
from PIL import Image


class RecurrentAttentionModel(object):
    def __init__(self, sess=None, batch_size=64, multi_batch_size=16, pth_size=64, variance=0.22, drop1=0.3, drop2=0.3,
                 internal1=512, internal2=512, internal3=512, decay=0.99, mode='new', l2_rate=0.1,
                 train_method='Adam', reinforce_mode='baseline', learning_rate=5e-4, min_learning_rate=1e-5,
                 resize_side_min=224, resize_side_max=512, origin_image_size=224, resize_image_size=224,
                 train_path='', valid_path='', test_path='', logging_step=50, train_epoch=20, test_period=5):

        self.batch_size = batch_size
        self.multi_batch_size = multi_batch_size

        self.train_step_per_epoch = (5094 // self.batch_size) if 5094 % self.batch_size == 0 else (5094 // self.batch_size + 1)
        self.test_step_per_epoch = (5794 // self.batch_size) if 5794 % self.batch_size == 0 else (5794 // self.batch_size + 1)
        self.test_multi_step_per_epoch = (5794 // self.multi_batch_size) if 5794 % self.multi_batch_size == 0 else (5794 // self.multi_batch_size + 1)
        self.valid_step_per_epoch = (900 // self.batch_size) if 900 % self.batch_size == 0 else (900 // self.batch_size + 1)

        self.origin_image_size = origin_image_size
        self.resize_image_size = resize_image_size
        self.pth_size = pth_size

        self.internal1 = internal1
        self.internal2 = internal2
        self.internal3 = internal3

        self.variance = variance

        self.drop1 = drop1
        self.drop2 = drop2
        self.decay = decay

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.learning_rate = learning_rate
        self.decay_learning_rate = tf.maximum(tf.train.exponential_decay(
            learning_rate, self.global_step,
            self.train_step_per_epoch,
            self.decay,
            staircase=True),
            min_learning_rate)

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

        self.round = sys.argv[1]
        self.mode = mode

        self.l2_rate = l2_rate

        self.train_method = train_method
        self.pth_mode = 'fix'
        self.reinforce_mode = reinforce_mode

        self.test_period = test_period

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
        scale = tf.cond(tf.greater(height, width), lambda: smallest_side / width, lambda: smallest_side / height)
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
            crop = tf.random_crop(image, [self.origin_image_size, self.origin_image_size, 3])
            image = tf.to_float(crop)
        normalize_image = self._mean_image_subtraction(image, [123.68, 116.78, 103.94])
        normalize_image = tf.image.random_flip_left_right(normalize_image)
        return normalize_image, label

    def _test_parse_func(self, filename, label):
        image_string = tf.read_file(filename)
        image = tf.image.decode_jpeg(image_string, channels=3)
        if self.mode == 'origin':
            image = tf.image.resize_images(image, [448, 448])
        else:
            image = self._aspect_preserving_resize(image, self.resize_side_min)
            crop = tf.image.resize_image_with_crop_or_pad(image, self.origin_image_size, self.origin_image_size)
            image = tf.to_float(crop)
        normalize_image = self._mean_image_subtraction(image, [123.68, 116.78, 103.94])
        return normalize_image, label

    def _test_multi_parse_func(self, filename, label):
        image_string = tf.read_file(filename)
        image = tf.image.decode_jpeg(image_string, channels=3)
        image = self._aspect_preserving_resize(image, self.resize_side_min)
        center_crop = tf.image.resize_image_with_crop_or_pad(image, self.origin_image_size, self.origin_image_size)
        # multi crop
        height, width = tf.shape(image)[0], tf.shape(image)[1]
        top_left_crop = tf.image.crop_to_bounding_box(image, 0, 0, self.origin_image_size, self.origin_image_size)
        top_right_crop = tf.image.crop_to_bounding_box(image, 0, width-self.origin_image_size, self.origin_image_size, self.origin_image_size)
        bottom_left_crop = tf.image.crop_to_bounding_box(image, height-self.origin_image_size, 0, self.origin_image_size, self.origin_image_size)
        bottom_right_crop = tf.image.crop_to_bounding_box(image, height-self.origin_image_size, width-self.origin_image_size, self.origin_image_size, self.origin_image_size)
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

    def resnet_feature(self, images, scope_name, train_mode=True):
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            net, end_points = resnet_v1.resnet_v1_50(images, 1000, is_training=train_mode, reuse=tf.AUTO_REUSE)
        resnet_block_4 = end_points[scope_name + '/resnet_v1_50/block4']
        resnet_feature = tf.reduce_mean(resnet_block_4, [1, 2], keepdims=True)
        resnet_feature = tf.squeeze(resnet_feature)
        resnet_feature = tf.reshape(resnet_feature, [-1, 2048])
        return resnet_block_4, resnet_feature

    def feature_extractor_1(self, images, train_mode=True):
        with tf.variable_scope('extractor_1'):
            resnet_block_4_1, resnet_feature_1 = self.resnet_feature(images, 'extractor_1', train_mode)
            drop1 = tf.layers.dropout(resnet_feature_1, rate=self.drop1, training=train_mode)
            fc1 = tf.layers.dense(inputs=drop1, units=self.internal1, activation=tf.nn.relu,
                                  kernel_initializer=tf.glorot_uniform_initializer(),
                                  bias_initializer=tf.constant_initializer(0.1),
                                  name='fc', reuse=tf.AUTO_REUSE)
            drop2 = tf.layers.dropout(fc1, rate=self.drop2, training=train_mode)
            logits_1 = tf.layers.dense(inputs=drop2, units=200,
                                  kernel_initializer=tf.glorot_uniform_initializer(),
                                  bias_initializer=tf.constant_initializer(0.1),
                                  name='logit', reuse=tf.AUTO_REUSE)
            loc_1 = tf.layers.dense(inputs=drop2, units=2,
                                    kernel_initializer=tf.glorot_uniform_initializer(),
                                    # kernel_initializer=tf.constant_initializer(0.0),
                                    bias_initializer=tf.constant_initializer(0.1),
                                    # kernel_regularizer=tf.contrib.layers.l1_regularizer(scale=1.0),
                                    name='loc', reuse=tf.AUTO_REUSE)
            loc_2 = tf.layers.dense(inputs=drop2, units=2,
                                    kernel_initializer=tf.glorot_uniform_initializer(),
                                    # kernel_initializer=tf.constant_initializer(0.0),
                                    bias_initializer=tf.constant_initializer(0.1),
                                    # kernel_regularizer=tf.contrib.layers.l1_regularizer(scale=1.0),
                                    name='loc_2', reuse=tf.AUTO_REUSE)
            loc_1 = loc_1 / 7.0
            loc_2 = loc_2 / 7.0
            scale_1 = tf.layers.dense(inputs=drop2, units=1,
                                  kernel_initializer=tf.glorot_uniform_initializer(),
                                  bias_initializer=tf.constant_initializer(0.1),
                                  # kernel_regularizer=tf.nn.l2_loss,
                                  name='scale', reuse=tf.AUTO_REUSE)
            scale_2 = tf.layers.dense(inputs=drop2, units=1,
                                  kernel_initializer=tf.glorot_uniform_initializer(),
                                  bias_initializer=tf.constant_initializer(0.1),
                                  # kernel_regularizer=tf.nn.l2_loss,
                                  name='scale_2', reuse=tf.AUTO_REUSE)
            scale_1 = (scale_1 / 10.0 + 1.0) / 2.0
            scale_2 = (scale_2 / 10.0 + 1.0) / 2.0
        return resnet_block_4_1, fc1, logits_1, loc_1, loc_2, scale_1, scale_2

    def feature_extractor_2(self, images, train_mode=True):
        with tf.variable_scope('extractor_2'):
            resnet_block_4_2, resnet_feature_2 = self.resnet_feature(images, 'extractor_2', train_mode)
            drop1 = tf.layers.dropout(resnet_feature_2, rate=self.drop1, training=train_mode)
            fc1 = tf.layers.dense(inputs=drop1, units=self.internal2, activation=tf.nn.relu,
                                  kernel_initializer=tf.glorot_uniform_initializer(),
                                  bias_initializer=tf.constant_initializer(0.1),
                                  name='fc', reuse=tf.AUTO_REUSE)
            drop2 = tf.layers.dropout(fc1, rate=self.drop2, training=train_mode)
            logits_2 = tf.layers.dense(inputs=drop2, units=200,
                                  kernel_initializer=tf.glorot_uniform_initializer(),
                                  bias_initializer=tf.constant_initializer(0.1),
                                  name='logit', reuse=tf.AUTO_REUSE)
            loc_2 = tf.layers.dense(inputs=drop2, units=2,
                                  kernel_initializer=tf.glorot_uniform_initializer(),
                                  bias_initializer=tf.constant_initializer(0.1),
                                  # kernel_regularizer=tf.nn.l2_loss,
                                  name='loc', reuse=tf.AUTO_REUSE)
            scale_2 = tf.layers.dense(inputs=drop2, units=1,
                                  kernel_initializer=tf.glorot_uniform_initializer(),
                                  bias_initializer=tf.constant_initializer(0.1),
                                  # kernel_regularizer=tf.nn.l2_loss,
                                  name='scale', reuse=tf.AUTO_REUSE)
        return resnet_block_4_2, logits_2, loc_2, scale_2

    def feature_extractor_3(self, images, train_mode=True):
        with tf.variable_scope('extractor_3'):
            resnet_block_4_3, resnet_feature = self.resnet_feature(images, 'extractor_3', train_mode)
            drop1 = tf.layers.dropout(resnet_feature, rate=self.drop1, training=train_mode)
            fc1 = tf.layers.dense(inputs=drop1, units=self.internal3, activation=tf.nn.relu,
                                  kernel_initializer=tf.glorot_uniform_initializer(),
                                  bias_initializer=tf.constant_initializer(0.1),
                                  name='fc', reuse=tf.AUTO_REUSE)
            drop2 = tf.layers.dropout(fc1, rate=self.drop2, training=train_mode)
            logits_3 = tf.layers.dense(inputs=drop2, units=200,
                                  kernel_initializer=tf.glorot_uniform_initializer(),
                                  bias_initializer=tf.constant_initializer(0.1),
                                  name='logit', reuse=tf.AUTO_REUSE)
        return logits_3

    def location_network(self, mean, variance, is_sampling):
        loc_mean = tf.clip_by_value(mean, -1., 1.)
        loc = tf.cond(is_sampling, lambda: loc_mean+tf.random_normal(tf.shape(loc_mean), stddev=variance), lambda: loc_mean)
        loc = tf.clip_by_value(loc, -1., 1.)
        loc = tf.stop_gradient(loc)
        log_prob = Normal(loc_mean, variance).log_prob(loc)
        log_prob = tf.reduce_sum(log_prob, -1)
        return loc, mean, log_prob

    def scale_network(self, mean, variance, is_sampling):
        scale_mean = tf.clip_by_value(mean, 0., 1.)
        scale = tf.cond(is_sampling, lambda: scale_mean+tf.random_normal(tf.shape(scale_mean), stddev=variance), lambda: scale_mean)
        scale = tf.clip_by_value(scale, 0., 1.)
        scale = tf.stop_gradient(scale)
        log_prob = Normal(scale_mean, variance).log_prob(scale)
        log_prob = tf.reduce_sum(log_prob, -1)
        return scale, mean, log_prob

    def location_baseline_network(self, logits):
        baseline = tf.layers.dense(logits, 1, kernel_initializer=tf.constant_initializer(0.0),
                                   bias_initializer=tf.constant_initializer(0.1),
                                   name='location/baseline/fc', reuse=tf.AUTO_REUSE)
        return tf.reshape(baseline, [-1])

    def scale_baseline_network(self, logits):
        baseline = tf.layers.dense(logits, 1, kernel_initializer=tf.constant_initializer(0.0),
                                   bias_initializer=tf.constant_initializer(0.1),
                                   name='scale/baseline/fc', reuse=tf.AUTO_REUSE)
        return tf.reshape(baseline, [-1])

    def generate_crop_box(self, loc, scale):
        normalize_loc = (loc + 1.0) / 2
        normalize_scale = scale / 2
        x_axis, y_axis = tf.split(normalize_loc, [1, 1], axis=-1)
        x_left = tf.clip_by_value(x_axis - normalize_scale, 0.0, 1.0)
        x_right = tf.clip_by_value(x_axis + normalize_scale, 0.0, 1.0)
        y_left = tf.clip_by_value(y_axis - normalize_scale, 0.0, 1.0)
        y_right = tf.clip_by_value(y_axis + normalize_scale, 0.0, 1.0)
        return tf.reshape(tf.stack([y_left, x_left, y_right, x_right], axis=-1), [-1, 4])

    def glimpse_extractor_1(self, img, argmax_loc, loc, scale):
        if self.round == '31' or self.round == '32':
            pth = tf.image.extract_glimpse(img, [self.pth_size, self.pth_size], argmax_loc)
            pth = tf.image.resize_images(pth, [self.resize_image_size, self.resize_image_size])
        elif self.round == '33':
            pth = tf.image.extract_glimpse(img, [self.pth_size, self.pth_size], loc)
            pth = tf.image.resize_images(pth, [self.resize_image_size, self.resize_image_size])
        else:
            crop_box = self.generate_crop_box(loc, scale)
            pth = tf.image.crop_and_resize(img, crop_box, tf.range(tf.shape(img)[0]), [self.resize_image_size, self.resize_image_size])
        return pth

    def glimpse_extractor_2(self, img, argmax_loc, loc, scale):
        if self.round == '41' or self.round == '42':
            pth = tf.image.extract_glimpse(img, [self.pth_size, self.pth_size], argmax_loc)
            pth = tf.image.resize_images(pth, [self.resize_image_size, self.resize_image_size])
        elif self.round == '43':
            pth = tf.image.extract_glimpse(img, [self.pth_size, self.pth_size], loc)
            pth = tf.image.resize_images(pth, [self.resize_image_size, self.resize_image_size])
        else:
            crop_box = self.generate_crop_box(loc, scale)
            pth = tf.image.crop_and_resize(img, crop_box, tf.range(tf.shape(img)[0]), [self.resize_image_size, self.resize_image_size])
        return pth

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

        resnet_feature_1, fc_1, logit_1, loc_mean_1, loc_mean_2, scale_1, scale_2 = self.feature_extractor_1(self.images, self.train_mode)
        self.loc_1, self.loc_mean_1, self.log_loc_prob_1 = self.location_network(loc_mean_1, self.variance, self.train_mode)
        self.scale_1, self.scale_mean_1, self.log_scale_prob_1 = self.scale_network(scale_1, self.variance, self.train_mode)

        self.prob1 = tf.nn.softmax(logit_1)
        acc1 = tf.cast(tf.equal(tf.argmax(self.prob1, axis=-1, output_type=tf.int32), self.labels), tf.float32)
        self.cls_loss_1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit_1, labels=self.labels))
        self.acc_1 = tf.reduce_mean(acc1)

        argmax_1 = tf.argmax(tf.reshape(tf.reduce_sum(tf.abs(resnet_feature_1), axis=-1), [tf.shape(resnet_feature_1)[0], -1]), axis=1)
        width, height = tf.shape(resnet_feature_1, out_type=tf.int64)[1], tf.shape(resnet_feature_1, out_type=tf.int64)[2]
        argmax_x_1 = argmax_1 // height / width
        argmax_y_1 = argmax_1 % height / height
        argmax_x_1 = argmax_x_1 * 2.0 - 1.0 + 1 / width
        argmax_y_1 = argmax_y_1 * 2.0 - 1.0 + 1 / height
        argmax_loc = tf.cast(tf.stack([argmax_x_1, argmax_y_1], axis=1), tf.float32)

        if self.round[0] == '4':
            self.glimpse_1 = self.glimpse_extractor_1(self.images, argmax_loc, self.loc_mean_1, self.scale_mean_1)
        else:
            self.glimpse_1 = self.glimpse_extractor_1(self.images, argmax_loc, self.loc_1, self.scale_1)

        ___, logit_2, _, __ = self.feature_extractor_2(self.glimpse_1, self.train_mode)
        self.prob2 = tf.nn.softmax(logit_2)
        acc2 = tf.cast(tf.equal(tf.argmax(self.prob2, axis=-1, output_type=tf.int32), self.labels), tf.float32)
        self.cls_loss_2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit_2, labels=self.labels))
        self.loc_loss_2 = -tf.reduce_mean(self.log_loc_prob_1 * tf.stop_gradient(acc2))
        self.scale_loss_2 = -tf.reduce_mean(self.log_scale_prob_1 * tf.stop_gradient(acc2))

        self.acc_2 = tf.reduce_mean(acc2)
        self.acc_12_1 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.prob1 + 0.1 * self.prob2, axis=-1, output_type=tf.int32), self.labels), tf.float32))
        self.acc_12_2 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.prob1 + 0.3 * self.prob2, axis=-1, output_type=tf.int32), self.labels), tf.float32))
        self.acc_12_3 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.prob1 + 0.5 * self.prob2, axis=-1, output_type=tf.int32), self.labels), tf.float32))
        self.acc_12_4 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.prob1 + 0.7 * self.prob2, axis=-1, output_type=tf.int32), self.labels), tf.float32))
        self.acc_12_5 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.prob1 + self.prob2, axis=-1, output_type=tf.int32), self.labels), tf.float32))

        self.loc_2, self.loc_mean_2, self.log_loc_prob_2 = self.location_network(loc_mean_2, self.variance, self.train_mode)
        self.scale_2, self.scale_mean_2, self.log_scale_prob_2 = self.scale_network(scale_2, self.variance, self.train_mode)
        self.glimpse_2 = self.glimpse_extractor_2(self.images, argmax_loc, self.loc_2, self.scale_2)

        logit_3 = self.feature_extractor_3(self.glimpse_2, self.train_mode)
        self.prob3 = tf.nn.softmax(logit_3)
        acc3 = tf.cast(tf.equal(tf.argmax(self.prob3, axis=-1, output_type=tf.int32), self.labels), tf.float32)
        self.cls_loss_3 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit_3, labels=self.labels))
        self.loc_loss_3 = -tf.reduce_mean(self.log_loc_prob_2 * tf.stop_gradient(acc3))
        self.scale_loss_3 = -tf.reduce_mean(self.log_scale_prob_2 * tf.stop_gradient(acc3))
        self.loc_loss_23 = - float(self.l2_rate) * tf.reduce_mean(tf.square(self.loc_mean_1 - self.loc_mean_2))
        self.acc_3 = tf.reduce_mean(acc3)

        self.acc_23_1 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.prob2 + 0.1 * self.prob3, axis=-1, output_type=tf.int32), self.labels), tf.float32))
        self.acc_23_2 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.prob2 + 0.3 * self.prob3, axis=-1, output_type=tf.int32), self.labels), tf.float32))
        self.acc_23_3 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.prob2 + 0.5 * self.prob3, axis=-1, output_type=tf.int32), self.labels), tf.float32))
        self.acc_23_4 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.prob2 + 0.7 * self.prob3, axis=-1, output_type=tf.int32), self.labels), tf.float32))
        self.acc_23_5 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.prob2 + self.prob3, axis=-1, output_type=tf.int32), self.labels), tf.float32))

        self.acc_123_1 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.prob1 + 0.7 * self.prob2 + 0.7 * self.prob3, axis=-1, output_type=tf.int32), self.labels), tf.float32))
        self.acc_123_2 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.prob1 + 0.5 * self.prob2 + 0.5 * self.prob3, axis=-1, output_type=tf.int32), self.labels), tf.float32))
        self.acc_123_3 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.prob1 + 1.0 * self.prob2 + 1.0 * self.prob3, axis=-1, output_type=tf.int32), self.labels), tf.float32))
        self.acc_123_4 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.prob1 + 0.5 * self.prob2 + self.prob3, axis=-1, output_type=tf.int32), self.labels), tf.float32))
        self.acc_123_5 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(0.5 * self.prob1 + self.prob2 + self.prob3, axis=-1, output_type=tf.int32), self.labels), tf.float32))
        self.acc_123_6 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.prob1 + self.prob2 + 0.5 * self.prob3, axis=-1, output_type=tf.int32), self.labels), tf.float32))
        self.acc_123_7 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.prob1 + 0.7 * self.prob2 + 0.5 * self.prob3, axis=-1, output_type=tf.int32), self.labels), tf.float32))

        # self.glimpse_2 = self.glimpse_extractor(self.glimpse_1, self.loc_1, scale_1)
        # logit_3 = self.feature_extractor_3(self.glimpse_2, self.train_mode)
        # self.prob3 = tf.nn.softmax(logit_3)
        # acc3 = tf.cast(tf.equal(tf.argmax(self.prob3, axis=-1, output_type=tf.int32), self.labels), tf.float32)
        # self.cls_loss_3 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit_3, labels=self.labels))
        # self.loc_loss_3 = tf.reduce_mean(self.log_action_prob_2 * acc3)
        # self.acc_3 = tf.reduce_mean(acc3)
        # self.loss_3 = self.cls_loss_3 + self.loc_loss_3
        # self.total_prob = self.f1 * self.prob1 + self.f2 * self.prob2 + self.f3 * self.prob3
        # self.total_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.total_prob, axis=-1, output_type=tf.int32), self.labels), tf.float32))

        if self.round == '11':
            self.loss = self.cls_loss_1
            self.acc = [self.acc_1]
            params = [var for var in tf.trainable_variables() if 'resnet' not in var.name and 'extractor_1' in var.name]
            tf.summary.scalar('cls_loss_1', self.cls_loss_1)
            tf.summary.scalar('acc_1', self.acc_1)
        elif self.round == '12':
            self.loss = self.cls_loss_1
            self.acc = [self.acc_1]
            params = [var for var in tf.trainable_variables() if 'extractor_1' in var.name]
            tf.summary.scalar('cls_loss_1', self.cls_loss_1)
            tf.summary.scalar('acc_1', self.acc_1)
        elif self.round == '21':
            self.loss = self.cls_loss_2
            self.acc = [self.acc_2]
            params = [var for var in tf.trainable_variables() if 'extractor_2/logit' in var.name]
            tf.summary.scalar('cls_loss_2', self.cls_loss_2)
            tf.summary.scalar('acc_2', self.acc_2)
        elif self.round == '22':
            self.loss = self.cls_loss_2
            self.acc = [self.acc_2]
            params = [var for var in tf.trainable_variables() if 'extractor_2' in var.name]
            tf.summary.scalar('cls_loss_2', self.cls_loss_2)
            tf.summary.scalar('acc_2', self.acc_2)
        elif self.round == '23':
            self.loss = self.loc_loss_2
            self.acc = [self.acc_1, self.acc_2, self.acc_12_1, self.acc_12_2, self.acc_12_3, self.acc_12_4, self.acc_12_5]
            params = [var for var in tf.trainable_variables() if 'extractor_1/loc/' in var.name]
            tf.summary.scalar('loc_loss_2', self.loc_loss_2)
            tf.summary.scalar('cls_loss_2', self.cls_loss_2)
            tf.summary.scalar('acc_2', self.acc_2)
            tf.summary.histogram('loc_mean_1', self.loc_mean_1)
        elif self.round == '24':
            self.loss = self.cls_loss_2
            self.acc = [self.acc_1, self.acc_2, self.acc_12_1, self.acc_12_2, self.acc_12_3, self.acc_12_4, self.acc_12_5]
            params = [var for var in tf.trainable_variables() if 'extractor_2' in var.name]
            tf.summary.scalar('cls_loss_2', self.cls_loss_2)
            tf.summary.scalar('acc_2', self.acc_2)
            tf.summary.histogram('loc_mean_1', self.loc_mean_1)
        elif self.round == '31':
            self.loss = self.cls_loss_2
            self.acc = [self.acc_2]
            params = [var for var in tf.trainable_variables() if 'extractor_2/logit' in var.name]
            tf.summary.scalar('cls_loss_2', self.cls_loss_2)
            tf.summary.scalar('acc_2', self.acc_2)
        elif self.round == '32':
            self.loss = self.cls_loss_2
            self.acc = [self.acc_1, self.acc_2, self.acc_12_1, self.acc_12_2, self.acc_12_3, self.acc_12_4, self.acc_12_5]
            params = [var for var in tf.trainable_variables() if 'extractor_2' in var.name]
            tf.summary.scalar('cls_loss_2', self.cls_loss_2)
            tf.summary.scalar('acc_2', self.acc_2)
        elif self.round == '33':
            self.loss = self.loc_loss_2
            self.acc = [self.acc_1, self.acc_2, self.acc_12_1, self.acc_12_2, self.acc_12_3, self.acc_12_4, self.acc_12_5]
            params = [var for var in tf.trainable_variables() if 'extractor_1/loc/' in var.name]
            tf.summary.scalar('loc_loss_2', self.loc_loss_2)
            tf.summary.scalar('cls_loss_2', self.cls_loss_2)
            tf.summary.scalar('acc_2', self.acc_2)
            tf.summary.histogram('loc_mean_1', self.loc_mean_1)
        elif self.round == '34':
            self.loss = self.scale_loss_2
            self.acc = [self.acc_1, self.acc_2, self.acc_12_1, self.acc_12_2, self.acc_12_3, self.acc_12_4, self.acc_12_5]
            params = [var for var in tf.trainable_variables() if 'extractor_1/scale/' in var.name]
            tf.summary.scalar('scale_loss_2', self.scale_loss_2)
            tf.summary.scalar('cls_loss_2', self.cls_loss_2)
            tf.summary.scalar('acc_2', self.acc_2)
            tf.summary.histogram('scale_mean_1', self.scale_mean_1)
        elif self.round == '35':
            self.loss = self.cls_loss_2
            self.acc = [self.acc_1, self.acc_2, self.acc_12_1, self.acc_12_2, self.acc_12_3, self.acc_12_4, self.acc_12_5]
            params = [var for var in tf.trainable_variables() if 'extractor_2' in var.name]
            tf.summary.scalar('cls_loss_2', self.cls_loss_2)
            tf.summary.scalar('acc_2', self.acc_2)
        elif self.round == '36':
            self.loss = self.loc_loss_2
            self.acc = [self.acc_1, self.acc_2, self.acc_12_1, self.acc_12_2, self.acc_12_3, self.acc_12_4, self.acc_12_5]
            params = [var for var in tf.trainable_variables() if 'extractor_1/loc/' in var.name]
            tf.summary.scalar('loc_loss_2', self.loc_loss_2)
            tf.summary.scalar('cls_loss_2', self.cls_loss_2)
            tf.summary.scalar('acc_2', self.acc_2)
            tf.summary.histogram('loc_mean_1', self.loc_mean_1)
        elif self.round == '41':
            self.loss = self.cls_loss_3
            self.acc = [self.acc_1, self.acc_2, self.acc_3]
            params = [var for var in tf.trainable_variables() if 'extractor_3/logit' in var.name]
            tf.summary.scalar('cls_loss_3', self.cls_loss_3)
            tf.summary.scalar('acc_3', self.acc_3)
        elif self.round == '42':
            self.loss = self.cls_loss_3
            self.acc = [self.acc_1, self.acc_2, self.acc_3]
            params = [var for var in tf.trainable_variables() if 'extractor_3' in var.name]
            tf.summary.scalar('cls_loss_3', self.cls_loss_3)
            tf.summary.scalar('acc_3', self.acc_3)
        elif self.round == '43':
            self.loss = self.loc_loss_3 + self.loc_loss_23
            self.acc = [self.acc_1, self.acc_2, self.acc_3, self.acc_12_4, self.acc_12_5,
                        self.acc_23_2, self.acc_23_3, self.acc_23_4]
            params = [var for var in tf.trainable_variables() if 'extractor_1/loc_2' in var.name]
            tf.summary.scalar('loc_loss_3', self.loc_loss_3)
            tf.summary.scalar('cls_loss_3', self.cls_loss_3)
            tf.summary.scalar('acc_3', self.acc_3)
            tf.summary.histogram('loc_mean_1', self.loc_mean_1)
            tf.summary.histogram('loc_mean_2', self.loc_mean_2)
            tf.summary.histogram('loc_mean_diff', self.loc_mean_1 - self.loc_mean_2)
        elif self.round == '44':
            self.loss = self.scale_loss_3
            self.acc = [self.acc_1, self.acc_2, self.acc_3, self.acc_12_4, self.acc_12_5, self.acc_23_4, self.acc_23_5,
                        self.acc_123_1, self.acc_123_2, self.acc_123_3, self.acc_123_4, self.acc_123_5, self.acc_123_6,
                        self.acc_123_7]
            params = [var for var in tf.trainable_variables() if 'extractor_1/scale_2' in var.name]
            tf.summary.scalar('scale_loss_2', self.scale_loss_3)
            tf.summary.scalar('cls_loss_3', self.cls_loss_3)
            tf.summary.scalar('acc_3', self.acc_3)
            tf.summary.histogram('scale_mean_2', self.scale_mean_2)
        elif self.round == '45':
            self.loss = self.cls_loss_3
            self.acc = [self.acc_1, self.acc_2, self.acc_3, self.acc_12_4, self.acc_12_5, self.acc_23_4, self.acc_23_5,
                        self.acc_123_1, self.acc_123_2, self.acc_123_3, self.acc_123_4, self.acc_123_5, self.acc_123_6,
                        self.acc_123_7]
            params = [var for var in tf.trainable_variables() if 'extractor_3' in var.name]
            tf.summary.scalar('cls_loss_3', self.cls_loss_3)
            tf.summary.scalar('acc_3', self.acc_3)
        elif self.round == '46':
            self.loss = self.loc_loss_3 + self.loc_loss_23
            self.acc = [self.acc_1, self.acc_2, self.acc_3, self.acc_12_4, self.acc_12_5, self.acc_23_4, self.acc_23_5,
                        self.acc_123_1, self.acc_123_2, self.acc_123_3, self.acc_123_4, self.acc_123_5, self.acc_123_6,
                        self.acc_123_7]
            params = [var for var in tf.trainable_variables() if 'extractor_1/loc_2' in var.name]
            tf.summary.scalar('loc_loss_3', self.loc_loss_3)
            tf.summary.scalar('cls_loss_3', self.cls_loss_3)
            tf.summary.scalar('acc_3', self.acc_3)
            tf.summary.histogram('loc_mean_1', self.loc_mean_1)
            tf.summary.histogram('loc_mean_2', self.loc_mean_2)
            tf.summary.histogram('loc_mean_diff', self.loc_mean_1 - self.loc_mean_2)
        else:
            sys.exit(1)

        gradients = tf.gradients(self.loss, params)
        if self.train_method == 'Adam':
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.decay_learning_rate).\
                apply_gradients(zip(gradients, params), global_step=self.global_step)
        else:
            self.train_op = tf.train.MomentumOptimizer(self.decay_learning_rate, 0.9). \
                apply_gradients(zip(gradients, params), global_step=self.global_step)

        for param in params:
            tf.summary.histogram(param.name, param)
        self.merged = tf.summary.merge_all()

    def run(self):
        self.sess.run(tf.global_variables_initializer())
        if self.round == '11':
            var_list = [var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if
                        self.train_method not in var.name and 'resnet' in var.name and 'extractor_1' in var.name]
            restore = tf.train.Saver(var_list)
            restore.restore(self.sess, 'extractor_1/model.ckpt')
            var_list = [var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if
                        self.train_method not in var.name and 'resnet' in var.name and 'extractor_2' in var.name]
            restore = tf.train.Saver(var_list)
            restore.restore(self.sess, 'extractor_2/model.ckpt')
        elif self.round == '33':
            var_list = [var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if
                        self.train_method not in var.name and 'global_step' not in var.name and 'power' not in var.name
                        and 'loc_2' not in var.name and 'scale_2' not in var.name and 'extractor_3' not in var.name]
            restore = tf.train.Saver(var_list)
            restore.restore(self.sess, sys.argv[2])
        elif self.round == '41':
            var_list = [var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if
                        self.train_method not in var.name and 'resnet' in var.name and 'extractor_3' in var.name]
            restore = tf.train.Saver(var_list)
            restore.restore(self.sess, 'extractor_3/model.ckpt')
            var_list = [var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if self.train_method not in var.name
                        and 'global_step' not in var.name and 'power' not in var.name and 'extractor_3' not in var.name]
            restore2 = tf.train.Saver(var_list)
            restore2.restore(self.sess, sys.argv[2])
        else:
            var_list = [var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if self.train_method not in var.name
                        and 'global_step' not in var.name and 'power' not in var.name]
            restore = tf.train.Saver(var_list)
            restore.restore(self.sess, sys.argv[2])

        saver = tf.train.Saver(max_to_keep=100)
        cur_time = str(time.time())
        train_writer = tf.summary.FileWriter('log/ram2/' + self.round + '_' + str(self.learning_rate) + '_' + cur_time + '/board/train', self.sess.graph)
        test_writer = tf.summary.FileWriter('log/ram2/' + self.round + '_' + str(self.learning_rate) + '_' + cur_time + '/board/test', self.sess.graph)
        valid_writer = tf.summary.FileWriter('log/ram2/' + self.round + '_' + str(self.learning_rate) + '_' + cur_time + '/board/valid', self.sess.graph)

        logging.info(self.mode)
        logging.info('{} {} {}'.format(self.learning_rate, self.train_method, self.decay))
        logging.info(cur_time)
        logging.info('{} {} {}'.format(self.pth_mode, self.reinforce_mode, self.l2_rate))
        logging.info('{} {} {} {} {} {} {} {} {} {}'.format(self.drop1, self.drop2, self.pth_size, self.internal1,
                                                      self.internal2, self.internal3, self.resize_side_min,
                                                      self.resize_side_max, self.origin_image_size, self.resize_image_size))

        # if self.round == 'check':
        #     self.sess.run(self.test_init_op)
        #     imgs, glimpses = self.sess.run([self.images, self.glimpse_1], feed_dict={self.train_mode: False})
        #     for i in range(8):
        #         image = Image.fromarray(imgs[i].astype('uint8'), 'RGB')
        #         glimpse = Image.fromarray(glimpses[i].astype('uint8'), 'RGB')
        #         image.show(title=str(i)+'image')
        #         glimpse.show(title=str(i)+'glimpse')
        #     sys.exit(1)


        for epoch in range(self.train_epoch):
            statistics = []
            self.sess.run(self.train_init_op)
            for train_step in range(self.train_step_per_epoch):
                _, summary, loss, acc = self.sess.run([self.train_op, self.merged, self.loss, self.acc],
                                                      feed_dict={self.train_mode: True})
                train_writer.add_summary(summary, epoch * self.train_step_per_epoch + train_step)
                statistics.append([loss] + acc)

                if train_step and train_step % self.logging_step == 0:
                    statistics = np.mean(np.asarray(statistics, np.float32), axis=0)
                    logging.info('Epoch {} Train step {}'.format(epoch, train_step))
                    logging.info(statistics)
                    statistics = []

            self.sess.run(self.valid_init_op)
            statistics = []
            for valid_step in range(self.valid_step_per_epoch):
                summary, loss, acc, = self.sess.run([self.merged, self.loss, self.acc],
                                                    feed_dict={self.train_mode: False})
                valid_writer.add_summary(summary, epoch * self.valid_step_per_epoch + valid_step)
                statistics.append([loss] + acc)
            statistics = np.mean(np.asarray(statistics, np.float32), axis=0)
            logging.info('Epoch {} Valid'.format(epoch))
            logging.info(statistics)

            if epoch % self.test_period == 0:
                self.sess.run(self.test_init_op)
                statistics = []
                for test_step in range(self.test_step_per_epoch):
                    summary, loss, acc = self.sess.run([self.merged, self.loss, self.acc],
                                                       feed_dict={self.train_mode: False})
                    test_writer.add_summary(summary, (epoch / self.test_period) * self.test_step_per_epoch + test_step)
                    statistics.append([loss] + acc)
                statistics = np.mean(np.asarray(statistics, np.float32), axis=0)
                logging.info('Epoch {} Test'.format(epoch))
                logging.info(statistics)

                saver.save(self.sess, 'log/ram2/' + self.round + '_' + str(self.learning_rate) + '_' + cur_time +
                           '/tmp/model.ckpt', global_step=epoch)

        train_writer.close()
        test_writer.close()
        valid_writer.close()


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    with tf.Session() as sess:
        model = RecurrentAttentionModel(sess, **config.recurrent_attention_model_2)
        model.run()
