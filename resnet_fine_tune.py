from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import sys
import time
import config
from tensorflow.contrib import slim
from tensorflow.contrib.slim.nets import resnet_v1


class ResNetFineTuneModel:
    def __init__(self, sess=None, batch_size=32, learning_rate=1e-4, resize_side_min=224, resize_side_max=512,
                 train_path='', valid_path='', test_path='', train_epoch=20, train_step_per_epoch=200,
                 test_step_per_epoch=100, logging_step=50):

        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.resize_side_min = resize_side_min
        self.resize_side_max = resize_side_max

        self.train_path = train_path
        self.valid_path = valid_path
        self.test_path = test_path

        self.train_image_names, self.train_labels = self._generate_image_names_and_labels(train_path)
        self.test_image_names, self.test_labels = self._generate_image_names_and_labels(test_path)

        self.train_epoch = train_epoch
        self.train_step_per_epoch = train_step_per_epoch
        self.test_step_per_epoch = test_step_per_epoch
        self.logging_step = logging_step

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
        image = tf.to_float(crop)
        normalize_image = self._mean_image_subtraction(image, [123.68, 116.78, 103.94])
        return normalize_image, label

    def _generate_train_image_label(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.train_image_names, self.train_labels))
        dataset = dataset.map(self._train_parse_func)
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.repeat()
        return dataset

    def generate_test_image_label(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.test_image_names, self.test_labels))
        dataset = dataset.map(self._test_parse_func)
        dataset = dataset.batch(self.batch_size)
        return dataset

    def feature_extractor(self, images, train_mode=True):
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            net, end_points = resnet_v1.resnet_v1_50(images, 200, is_training=train_mode, reuse=tf.AUTO_REUSE)
        resnet_feature = end_points['resnet_v1_50/block4']
        resnet_feature = tf.reduce_mean(resnet_feature, [1, 2], keepdims=True)
        resnet_feature = tf.squeeze(resnet_feature)
        resnet_feature = tf.reshape(resnet_feature, [-1, 2048])
        drop1 = tf.layers.dropout(resnet_feature, rate=0.5, training=train_mode)
        fc1 = tf.layers.dense(inputs=drop1, units=512, activation=tf.nn.relu,
                              kernel_initializer=tf.glorot_uniform_initializer(),
                              bias_initializer=tf.constant_initializer(0.1), kernel_regularizer=tf.nn.l2_loss,
                              name='extractor/fc/1', reuse=tf.AUTO_REUSE)
        drop2 = tf.layers.dropout(fc1, rate=0.3, training=train_mode)
        fc2 = tf.layers.dense(inputs=drop2, units=200,
                              kernel_initializer=tf.glorot_uniform_initializer(),
                              bias_initializer=tf.constant_initializer(0.1), kernel_regularizer=tf.nn.l2_loss,
                              name='extractor/fc/2', reuse=tf.AUTO_REUSE)
        return fc2

    def model(self):
        self.train_mode = tf.placeholder(tf.bool)
        self.train_dataset = self._generate_train_image_label()
        self.test_dataset = self.generate_test_image_label()

        self.iter = tf.data.Iterator.from_structure(self.train_dataset.output_types, self.train_dataset.output_shapes)
        self.train_init_op = self.iter.make_initializer(self.train_dataset)
        self.test_init_op = self.iter.make_initializer(self.test_dataset)

        self.images, self.labels = self.iter.get_next()

        self.logits = self.feature_extractor(self.images, self.train_mode)
        self.probs = tf.nn.softmax(self.logits)
        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels))
        self.predict_labels = tf.argmax(self.logits, axis=-1, output_type=tf.int32)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predict_labels, self.labels), tf.float32))
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        tf.summary.histogram('probs', self.probs)
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('accuracy', self.accuracy)
        self.merged = tf.summary.merge_all()

    def run(self):
        self.sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        cur_time = str(time.time())
        train_writer = tf.summary.FileWriter('run/resnet_fine_tune/' + str(self.learning_rate) + '_' + str(cur_time) + '/board/train', sess.graph)
        test_writer = tf.summary.FileWriter('run/resnet_fine_tune/' + str(self.learning_rate) + '_' + str(cur_time) + '/board/test', sess.graph)

        sys.stdout = open('run/resnet_fine_tune/' + str(self.learning_rate) + '_' + str(cur_time) + '.txt', 'wt')
        print(str(self.learning_rate))
        print(cur_time)

        for epoch in range(self.train_epoch):
            self.sess.run(self.train_init_op)
            for train_step in range(self.train_step_per_epoch):
                __, summary, loss, accuracy = self.sess.run([self.train_op, self.merged, self.loss, self.accuracy],
                                                            feed_dict={self.train_mode: True})

                train_writer.add_summary(summary, epoch * self.train_step_per_epoch + train_step)

                if train_step and train_step % self.logging_step == 0:
                    print('Epoch {} Train step {}: loss = {:3.4f}\t accuracy = {:3.4f}'
                          .format(epoch, train_step, loss, accuracy))

            saver.save(sess, 'run/resnet_fine_tune/' + str(self.learning_rate) + '_' + str(cur_time) + '/tmp/model.ckpt', global_step=epoch)
            self.sess.run(self.test_init_op)
            losses, accs = [], []
            for test_step in range(self.test_step_per_epoch):
                summary, loss, accuracy = self.sess.run([self.merged, self.loss, self.accuracy],
                                                        feed_dict={self.train_mode: False})
                test_writer.add_summary(summary, epoch * self.test_step_per_epoch + test_step)
                losses.append(loss)
                accs.append(accuracy)
            print('Epoch {}: loss = {:3.4f}\t accuracy = {:3.4f}'
                  .format(epoch, np.mean(losses), np.mean(accs)))

        train_writer.close()
        test_writer.close()


if __name__ == "__main__":
    with tf.Session() as sess:
        fine_tune_model = ResNetFineTuneModel(sess, **config.resnet_fine_tune_model)
        fine_tune_model.run()
