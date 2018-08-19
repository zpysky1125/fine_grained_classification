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
from tensorflow.python.ops.distributions.normal import Normal
from tensorflow.contrib import slim
from tensorflow.contrib.slim.nets import resnet_v1


class RecurrentAttentionModel(object):
    def __init__(self, sess=None, batch_size=64, multi_batch_size=16, pth_size=64, variance=0.22, drop1=0.3, drop2=0.3,
                 internal1=512, internal2=512, internal3=512,
                 train_method='Adam', crop_or_mask='crop', reinforce_mode='baseline',
                 learning_rate=5e-4, min_learning_rate=1e-5, resize_side_min=224, resize_side_max=512,
                 train_path='', valid_path='', test_path='', logging_step=50, train_epoch=20):

        self.batch_size = batch_size
        self.multi_batch_size = multi_batch_size

        self.train_step_per_epoch = (5094 // self.batch_size) if 5094 % self.batch_size == 0 else (5094 // self.batch_size + 1)
        self.test_step_per_epoch = (5794 // self.batch_size) if 5794 % self.batch_size == 0 else (5794 // self.batch_size + 1)
        self.test_multi_step_per_epoch = (5794 // self.multi_batch_size) if 5794 % self.multi_batch_size == 0 else (5794 // self.multi_batch_size + 1)
        self.valid_step_per_epoch = (900 // self.batch_size) if 900 % self.batch_size == 0 else (900 // self.batch_size + 1)

        self.img_size = 224
        self.pth_size = pth_size

        self.internal1 = internal1
        self.internal2 = internal2
        self.internal3 = internal3

        self.variance = variance

        self.drop1 = drop1
        self.drop2 = drop2

        self.f1 = 1.0
        self.f2 = 1.0
        self.f3 = 1.0

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.learning_rate = learning_rate
        self.decay_learning_rate = tf.maximum(tf.train.exponential_decay(
            learning_rate, self.global_step,
            self.train_step_per_epoch,
            0.99,
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
        self.mode = 'init'

        self.train_method = train_method
        self.crop_or_mask = crop_or_mask
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

    def resnet_feature(self, images, train_mode=True):
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            net, end_points = resnet_v1.resnet_v1_50(images, 1000, is_training=train_mode, reuse=tf.AUTO_REUSE)
        resnet_feature = end_points['resnet_v1_50/block4']
        resnet_feature = tf.reduce_mean(resnet_feature, [1, 2], keepdims=True)
        resnet_feature = tf.squeeze(resnet_feature)
        resnet_feature = tf.reshape(resnet_feature, [-1, 2048])
        return resnet_feature

    def feature_extractor_1(self, resnet_feature, train_mode=True):
        drop1 = tf.layers.dropout(resnet_feature, rate=self.drop1, training=train_mode)
        fc1 = tf.layers.dense(inputs=drop1, units=self.internal1, activation=tf.nn.relu,
                              kernel_initializer=tf.glorot_uniform_initializer(),
                              bias_initializer=tf.constant_initializer(0.1),
                              # kernel_regularizer=tf.nn.l2_loss,
                              name='extractor_1/fc_1', reuse=tf.AUTO_REUSE)
        drop2 = tf.layers.dropout(fc1, rate=self.drop2, training=train_mode)
        fc2 = tf.layers.dense(inputs=drop2, units=203,
                              kernel_initializer=tf.glorot_uniform_initializer(),
                              bias_initializer=tf.constant_initializer(0.1),
                              # kernel_regularizer=tf.nn.l2_loss,
                              name='extractor_1/fc_2', reuse=tf.AUTO_REUSE)
        return tf.split(fc2, [200, 2, 1], axis=-1)

    def feature_extractor_2(self, resnet_feature, train_mode=True):
        drop1 = tf.layers.dropout(resnet_feature, rate=self.drop1, training=train_mode)
        fc1 = tf.layers.dense(inputs=drop1, units=self.internal2, activation=tf.nn.relu,
                              kernel_initializer=tf.glorot_uniform_initializer(),
                              bias_initializer=tf.constant_initializer(0.1),
                              # kernel_regularizer=tf.nn.l2_loss,
                              name='extractor_2/fc_1', reuse=tf.AUTO_REUSE)
        drop2 = tf.layers.dropout(fc1, rate=self.drop2, training=train_mode)
        fc2 = tf.layers.dense(inputs=drop2, units=203,
                              kernel_initializer=tf.glorot_uniform_initializer(),
                              bias_initializer=tf.constant_initializer(0.1),
                              # kernel_regularizer=tf.nn.l2_loss,
                              name='extractor_2/fc_2', reuse=tf.AUTO_REUSE)
        return tf.split(fc2, [200, 2, 1], axis=-1)

    def feature_extractor_3(self, resnet_feature, train_mode=True):
        drop1 = tf.layers.dropout(resnet_feature, rate=self.drop1, training=train_mode)
        fc1 = tf.layers.dense(inputs=drop1, units=self.internal3, activation=tf.nn.relu,
                              kernel_initializer=tf.glorot_uniform_initializer(),
                              bias_initializer=tf.constant_initializer(0.1),
                              # kernel_regularizer=tf.nn.l2_loss,
                              name='extractor_3/fc_1', reuse=tf.AUTO_REUSE)
        drop2 = tf.layers.dropout(fc1, rate=self.drop2, training=train_mode)
        fc2 = tf.layers.dense(inputs=drop2, units=200,
                              kernel_initializer=tf.glorot_uniform_initializer(),
                              bias_initializer=tf.constant_initializer(0.1),
                              # kernel_regularizer=tf.nn.l2_loss,
                              name='extractor_3/fc_2', reuse=tf.AUTO_REUSE)
        return fc2

    def location_network(self, mean, variance, is_sampling):
        mean = tf.clip_by_value(mean, -1., 1.)
        loc = tf.cond(is_sampling, lambda: tf.clip_by_value(mean + tf.random_normal((tf.shape(mean)[0], 2),
                                                                stddev=self.variance), -1., 1.), lambda: mean)
        log_prob = Normal(mean, variance)._log_prob(x=loc)
        log_prob = tf.reduce_sum(log_prob, -1)
        return loc, mean, log_prob

    def baseline_network(self, rnn_outputs):
        baseline = tf.layers.dense(tf.stack(rnn_outputs), 1, kernel_initializer=tf.glorot_uniform_initializer(),
                                   bias_initializer=tf.constant_initializer(0.1),
                                   # kernel_regularizer=tf.nn.l2_loss,
                                   name='baseline/fc', reuse=tf.AUTO_REUSE)
        return baseline

    def glimpse_extractor(self, img, loc, size):
        ll = tf.stop_gradient(loc)
        pth = tf.image.extract_glimpse(img, [self.pth_size, self.pth_size], ll)
        if self.crop_or_mask == 'crop':
            pth = tf.image.resize_images(pth, [224, 224])
        else:
            pth = tf.image.pad_to_bounding_box(pth, int((224 - self.pth_size) / 2),
                                               int((224 - self.pth_size) / 2), 224, 224)
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

        resnet_features_1 = self.resnet_feature(self.images, self.train_mode)
        logit_1, loc_mean_1, scale_1 = self.feature_extractor_1(resnet_features_1, self.train_mode)
        self.loc_1, self.loc_mean_1, self.log_action_prob_1 = self.location_network(loc_mean_1, self.variance, self.train_mode)
        self.prob1 = tf.nn.softmax(logit_1)
        acc1 = tf.cast(tf.equal(tf.argmax(self.prob1, axis=-1, output_type=tf.int32), self.labels), tf.float32)
        self.cls_loss_1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit_1, labels=self.labels))
        self.glimpse_1 = self.glimpse_extractor(self.images, self.loc_1, scale_1)
        self.acc_1 = tf.reduce_mean(acc1)
        self.loss_1 = self.cls_loss_1


        resnet_features_2 = self.resnet_feature(self.glimpse_1, self.train_mode)
        logit_2, loc_mean_2, scale_2 = self.feature_extractor_2(resnet_features_2, self.train_mode)
        self.loc_2, self.loc_mean_2, self.log_action_prob_2 = self.location_network(loc_mean_2, self.variance, self.train_mode)
        self.prob2 = tf.nn.softmax(logit_2)
        acc2 = tf.cast(tf.equal(tf.argmax(self.prob2, axis=-1, output_type=tf.int32), self.labels), tf.float32)
        self.cls_loss_2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit_2, labels=self.labels))
        self.loc_loss_2 = tf.reduce_mean(self.log_action_prob_1 * acc2)
        self.acc_2 = tf.reduce_mean(acc2)
        self.loss_2 = self.cls_loss_2 + self.loc_loss_2
        self.glimpse_2 = self.glimpse_extractor(self.glimpse_1, self.loc_1, scale_1)

        resnet_features_3 = self.resnet_feature(self.glimpse_2, self.train_mode)
        logit_3 = self.feature_extractor_3(resnet_features_3, self.train_mode)
        self.prob3 = tf.nn.softmax(logit_3)
        acc3 = tf.cast(tf.equal(tf.argmax(self.prob3, axis=-1, output_type=tf.int32), self.labels), tf.float32)
        self.cls_loss_3 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit_3, labels=self.labels))
        self.loc_loss_3 = tf.reduce_mean(self.log_action_prob_2 * acc3)
        self.acc_3 = tf.reduce_mean(acc3)
        self.loss_3 = self.cls_loss_3 + self.loc_loss_3

        self.total_prob = self.f1 * self.prob1 + self.f2 * self.prob2 + self.f3 * self.prob3
        self.total_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.total_prob, axis=-1, output_type=tf.int32), self.labels), tf.float32))

        if self.round == '11':
            self.loss = self.loss_1
            self.acc = [self.acc_1]
            params = [var for var in tf.trainable_variables() if 'resnet' not in var.name]
            for param in params:
                print (param.name)
            tf.summary.scalar('cls_loss_1', self.cls_loss_1)
            tf.summary.scalar('acc_1', self.acc_1)
        elif self.round == '12':
            self.loss = self.loss_1
            self.acc = [self.acc_1]
            params = tf.trainable_variables()
            tf.summary.scalar('cls_loss_1', self.cls_loss_1)
            tf.summary.scalar('loss_1', self.cls_loss_2)
        elif self.round == '21':
            self.loss = self.loss_2
            self.acc = [self.acc_2]
            params = [var for var in tf.trainable_variables() if 'resnet' not in var.name and 'extractor_1/fc_1' not in var.name]
            tf.summary.scalar('cls_loss_2', self.cls_loss_2)
            tf.summary.scalar('loc_loss_2', self.loc_loss_2)
            tf.summary.scalar('acc_2', self.acc_2)
        elif self.round == '22':
            self.loss = self.loss_1 + self.loss_2
            self.acc = [self.acc_1, self.acc_2]
            params = tf.trainable_variables()
            tf.summary.scalar('cls_loss_1', self.cls_loss_1)
            tf.summary.scalar('acc_1', self.acc_1)
            tf.summary.scalar('cls_loss_2', self.cls_loss_2)
            tf.summary.scalar('loc_loss_2', self.loc_loss_2)
            tf.summary.scalar('acc_2', self.acc_2)
        elif self.round == '31':
            self.loss = self.loss_3
            self.acc = [self.acc_3]
            params = [var for var in tf.trainable_variables() if 'resnet' not in var.name and 'extractor_1'
                      not in var.name and 'extractor_2/fc_1' not in var.name]
            tf.summary.scalar('cls_loss_3', self.cls_loss_3)
            tf.summary.scalar('loc_loss_3', self.loc_loss_3)
            tf.summary.scalar('acc_3', self.acc_3)
        else:
            self.loss = self.loss_1 + self.loss_2 + self.loss_3
            self.acc = [self.acc_1, self.acc_2, self.acc_3]
            params = tf.trainable_variables()
            tf.summary.scalar('cls_loss_1', self.cls_loss_1)
            tf.summary.scalar('acc_1', self.acc_1)
            tf.summary.scalar('cls_loss_2', self.cls_loss_2)
            tf.summary.scalar('loc_loss_2', self.loc_loss_2)
            tf.summary.scalar('acc_2', self.acc_2)
            tf.summary.scalar('cls_loss_3', self.cls_loss_3)
            tf.summary.scalar('loc_loss_3', self.loc_loss_3)
            tf.summary.scalar('acc_3', self.acc_3)

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
            init_fn = slim.assign_from_checkpoint_fn('resnet_v1_50.ckpt', slim.get_model_variables('resnet_v1_50'))
            init_fn(self.sess)
        elif self.mode == '12':
            var_list = [var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if self.train_method not in var.name
                        and 'global_step' not in var.name]
            restore = tf.train.Saver(var_list)
            restore.restore(self.sess, sys.argv[2])
        elif self.mode == '21':
            var_list = [var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if self.train_method not in var.name
                        and 'global_step' not in var.name and 'extractor_2' not in var.name]
            restore = tf.train.Saver(var_list)
            restore.restore(self.sess, sys.argv[2])
        elif self.mode == '22':
            var_list = [var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if self.train_method not in var.name
                        and 'global_step' not in var.name]
            restore = tf.train.Saver(var_list)
            restore.restore(self.sess, sys.argv[2])
        elif self.mode == '31':
            var_list = [var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if self.train_method not in var.name
                        and 'global_step' not in var.name and 'extractor_3' not in var.name]
            restore = tf.train.Saver(var_list)
            restore.restore(self.sess, sys.argv[2])
        elif self.mode == '32':
            var_list = [var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if self.train_method not in var.name
                        and 'global_step' not in var.name]
            restore = tf.train.Saver(var_list)
            restore.restore(self.sess, sys.argv[2])
        else:
            sys.exit(1)

        saver = tf.train.Saver(max_to_keep=50)
        cur_time = str(time.time())
        train_writer = tf.summary.FileWriter('log/ram2/' + self.round + '_' + str(self.learning_rate) + '_' + cur_time + '/board/train', self.sess.graph)
        test_writer = tf.summary.FileWriter('log/ram2/' + self.round + '_' + str(self.learning_rate) + '_' + cur_time + '/board/test', self.sess.graph)
        valid_writer = tf.summary.FileWriter('log/ram2/' + self.round + '_' + str(self.learning_rate) + '_' + cur_time + '/board/valid', self.sess.graph)

        logging.info(self.mode)
        logging.info(str(self.learning_rate))
        logging.info(cur_time)
        logging.info(self.train_method)
        logging.info('{} {} {}'.format(self.crop_or_mask, self.pth_mode, self.reinforce_mode))
        logging.info('{} {} {} {} {} {} {} {}'.format(self.drop1, self.drop2, self.pth_size, self.internal1,
                                                      self.internal2, self.internal3, self.resize_side_min,
                                                      self.resize_side_max))

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

            if epoch and epoch % 5 == 0:
                self.sess.run(self.test_init_op)
                statistics = []
                for test_step in range(self.test_step_per_epoch):
                    summary, loss, acc = self.sess.run([self.merged, self.loss, self.acc],
                                                       feed_dict={self.train_mode: False})
                    test_writer.add_summary(summary, (epoch / 5) * self.test_step_per_epoch + test_step)
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
