from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import sys
import logging
import time
import config
from tensorflow.contrib import slim
from tensorflow.contrib.slim.nets import resnet_v1


class ResNetFineTuneModel:
    def __init__(self, sess=None, batch_size=32, learning_rate=1e-4, resize_side_min=224, resize_side_max=512,
                 drop1=0.3, drop2=0.3, internal_size=512, train_method='Adam',
                 train_path='', valid_path='', test_path='', train_epoch=20, train_step_per_epoch=200,
                 test_step_per_epoch=100, valid_step_per_epoch=100, logging_step=50):

        self.batch_size = batch_size

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.learning_rate = learning_rate
        self.decay_learning_rate = tf.maximum(tf.train.exponential_decay(
            learning_rate, self.global_step,
            train_step_per_epoch,
            0.99,
            staircase=True,
            name='learning_rate_decay'),
            3e-5)

        self.resize_side_min = resize_side_min
        self.resize_side_max = resize_side_max

        self.train_path = train_path
        self.valid_path = valid_path
        self.test_path = test_path

        self.train_image_names, self.train_labels = self._generate_image_names_and_labels(train_path)
        self.test_image_names, self.test_labels = self._generate_image_names_and_labels(test_path)
        self.valid_image_names, self.valid_labels = self._generate_image_names_and_labels(valid_path)

        self.train_epoch = train_epoch
        self.train_step_per_epoch = train_step_per_epoch
        self.test_step_per_epoch = test_step_per_epoch
        self.valid_step_per_epoch = valid_step_per_epoch

        self.logging_step = logging_step

        self.drop1 = drop1
        self.drop2 = drop2
        self.internal = internal_size
        self.train_method = train_method

        self.sess = sess

        self.mode = sys.argv[1]

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
        if image.get_shape().ndims != 3:
            raise ValueError('Input must be of size [height, width, C>0]')
        num_channels = image.get_shape().as_list()[-1]
        if len(means) != num_channels:
            raise ValueError('len(means) must match the number of channels')

        channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
        for i in range(num_channels):
            channels[i] -= means[i]
        return tf.concat(axis=2, values=channels)

    def _train_parse_func(self, filename, label):
        image_string = tf.read_file(filename)
        image = tf.image.decode_jpeg(image_string, channels=3)
        resize_side = tf.random_uniform([], minval=self.resize_side_min, maxval=self.resize_side_max + 1,
                                        dtype=tf.int32)
        image = self._aspect_preserving_resize(image, resize_side)
        crop = tf.random_crop(image, [224, 224, 3])
        image = tf.to_float(crop)
        normalize_image = self._mean_image_subtraction(image, [123.68, 116.78, 103.94])
        normalize_image = tf.image.random_flip_left_right(normalize_image)
        return normalize_image, label

    def _test_parse_func(self, filename, label):
        image_string = tf.read_file(filename)
        image = tf.image.decode_jpeg(image_string, channels=3)
        image = self._aspect_preserving_resize(image, self.resize_side_min)
        crop = tf.image.resize_image_with_crop_or_pad(image, 224, 224)
        # multi crop
        # height, width = tf.shape(image)[0], tf.shape(image)[1]
        # top_left_crop = tf.image.crop_to_bounding_box(image, 0, 0, 224, 224)
        # top_right_crop = tf.image.crop_to_bounding_box(image, 0, width-224, 224, 224)
        # bottom_left_crop = tf.image.crop_to_bounding_box(image, height-224, 0, 224, 224)
        # bottom_right_crop = tf.image.crop_to_bounding_box(image, height-224, width-224, 224, 224)

        image = tf.to_float(crop)
        normalize_image = self._mean_image_subtraction(image, [123.68, 116.78, 103.94])
        return normalize_image, label

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

    def feature_extractor(self, images, train_mode=True):
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            net, end_points = resnet_v1.resnet_v1_50(images, 1000, is_training=train_mode, reuse=tf.AUTO_REUSE)
        resnet_feature = end_points['resnet_v1_50/block4']
        resnet_feature = tf.reduce_mean(resnet_feature, [1, 2], keepdims=True)
        resnet_feature = tf.squeeze(resnet_feature)
        resnet_feature = tf.reshape(resnet_feature, [-1, 2048])
        drop1 = tf.layers.dropout(resnet_feature, rate=self.drop1, training=train_mode)
        fc1 = tf.layers.dense(inputs=drop1, units=self.internal, activation=tf.nn.relu,
                              kernel_initializer=tf.glorot_uniform_initializer(),
                              bias_initializer=tf.constant_initializer(0.1),
                              # kernel_regularizer=tf.nn.l2_loss,
                              name='glimpse_feature/fc', reuse=tf.AUTO_REUSE)
        drop2 = tf.layers.dropout(fc1, rate=self.drop2, training=train_mode)
        fc2 = tf.layers.dense(inputs=drop2, units=200,
                              kernel_initializer=tf.glorot_uniform_initializer(),
                              bias_initializer=tf.constant_initializer(0.1),
                              # kernel_regularizer=tf.nn.l2_loss,
                              name='extractor/fc', reuse=tf.AUTO_REUSE)
        return fc2

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

        self.logits = self.feature_extractor(self.images, self.train_mode)
        self.probs = tf.reduce_sum(tf.nn.softmax(self.logits) * tf.one_hot(self.labels, 200), axis=-1)
        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels))
        self.predict_labels = tf.argmax(self.logits, axis=-1, output_type=tf.int32)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predict_labels, self.labels), tf.float32))

        if self.mode == 'fine_tune':
            params = tf.trainable_variables()
        else:
            params = [param for param in tf.trainable_variables() if 'resnet' not in param.name]
        gradients = tf.gradients(self.loss, params)
        # clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
        # self.train_op = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(
        #     zip(clipped_gradients, params), global_step=self.global_step)

        if self.train_method == 'Adam':
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.decay_learning_rate).\
                apply_gradients(zip(gradients, params), global_step=self.global_step)
        else:
            self.train_op = tf.train.MomentumOptimizer(self.decay_learning_rate, 0.9). \
                apply_gradients(zip(gradients, params), global_step=self.global_step)

        for param in params:
            tf.summary.histogram(param.name, param)
        tf.summary.histogram('probs', self.probs)
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('accuracy', self.accuracy)
        self.merged = tf.summary.merge_all()

    def run(self):
        self.sess.run(tf.global_variables_initializer())
        if self.mode == 'fine_tune':
            var_list = [var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if 'Adam' not in var.name
                        and 'global_step' not in var.name and 'resnet' not in var.name]
            restore = tf.train.Saver(var_list)
            restore.restore(self.sess, sys.argv[2])
        elif self.mode == 'init':
            init_fn = slim.assign_from_checkpoint_fn('resnet_v1_50.ckpt', slim.get_model_variables('resnet_v1_50'))
            init_fn(self.sess)
        else:
            sys.exit(1)

        saver = tf.train.Saver(max_to_keep=10)
        cur_time = str(time.time())
        train_writer = tf.summary.FileWriter('log/resnet_fine_tune/' + self.mode + '_' + str(self.learning_rate) + '_' + str(cur_time) + '/board/train', sess.graph)
        test_writer = tf.summary.FileWriter('log/resnet_fine_tune/' + self.mode + '_' + str(self.learning_rate) + '_' + str(cur_time) + '/board/test', sess.graph)
        valid_writer = tf.summary.FileWriter('log/resnet_fine_tune/' + self.mode + '_' + str(self.learning_rate) + '_' + str(cur_time) + '/board/valid', sess.graph)

        logging.info(str(self.learning_rate))
        logging.info(cur_time)
        logging.info('{} {} {} {} {}'.format(self.drop1, self.drop2, self.internal, self.resize_side_min, self.resize_side_max))

        for epoch in range(self.train_epoch):
            self.sess.run(self.train_init_op)
            for train_step in range(self.train_step_per_epoch):
                __, summary, loss, accuracy = self.sess.run([self.train_op, self.merged, self.loss, self.accuracy],
                                                            feed_dict={self.train_mode: True})

                train_writer.add_summary(summary, epoch * self.train_step_per_epoch + train_step)

                if train_step and train_step % self.logging_step == 0:
                    logging.info('Epoch {} Train step {}: loss = {:3.4f}\t accuracy = {:3.4f}'
                                 .format(epoch, train_step, loss, accuracy))


            self.sess.run(self.valid_init_op)
            losses, accs = [], []
            for valid_step in range(self.valid_step_per_epoch):
                summary, loss, accuracy = self.sess.run([self.merged, self.loss, self.accuracy],
                                                        feed_dict={self.train_mode: False})
                valid_writer.add_summary(summary, epoch * self.valid_step_per_epoch + valid_step)
                losses.append(loss)
                accs.append(accuracy)
            logging.info('Epoch {} Valid: loss = {:3.4f}\t accuracy = {:3.4f}'.format(epoch, np.mean(losses), np.mean(accs)))

            if epoch and epoch % 5 == 0:
                self.sess.run(self.test_init_op)
                losses, accs = [], []
                for test_step in range(self.test_step_per_epoch):
                    summary, loss, accuracy = self.sess.run([self.merged, self.loss, self.accuracy],
                                                            feed_dict={self.train_mode: False})
                    test_writer.add_summary(summary, (epoch/5) * self.test_step_per_epoch + test_step)
                    losses.append(loss)
                    accs.append(accuracy)
                logging.info('Epoch {} Test: loss = {:3.4f}\t accuracy = {:3.4f}'.format(epoch, np.mean(losses), np.mean(accs)))
                saver.save(self.sess, 'log/resnet_fine_tune/' + self.mode + '_' + str(self.learning_rate) + '_' + str(
                    cur_time) + '/tmp/model.ckpt', global_step=epoch)

        train_writer.close()
        test_writer.close()
        valid_writer.close()


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    with tf.Session() as sess:
        fine_tune_model = ResNetFineTuneModel(sess, **config.resnet_fine_tune_model)
        fine_tune_model.run()
