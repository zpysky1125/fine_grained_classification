from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import logging
import os
import time
import config
from tensorflow.python.ops.rnn_cell_impl import BasicLSTMCell
from tensorflow.python.ops.distributions.normal import Normal
from tensorflow.contrib import slim
from tensorflow.contrib.slim.nets import resnet_v1
from PIL import Image


class RecurrentAttentionModel(object):
    def __init__(self, sess=None, batch_size=64, pth_size=64, glimpse_output_size=1024, cell_size=256, variance=0.22,
                 num_glimpses=5, glimpse_times=5,
                 learning_rate=5e-4, learning_rate_decay_factor=0.97, min_learning_rate=1e-5, max_gradient_norm=5.0,
                 resize_side_min=224, resize_side_max=512, train_path='', valid_path='', test_path='', logging_step=50,
                 train_epoch=20, train_step_per_epoch=200, test_step_per_epoch=100, valid_step_per_epoch=100):

        self.batch_size = batch_size

        self.img_size = 224
        self.pth_size = pth_size

        self.lstm_cell_size = cell_size
        self.glimpse_output_size = glimpse_output_size

        self.loc_dim = 2

        self.variance = variance

        self.num_glimpses = num_glimpses
        self.glimpse_times = glimpse_times

        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = learning_rate
        self.decay_learning_rate = tf.maximum(tf.train.exponential_decay(
            learning_rate, self.global_step,
            train_step_per_epoch,
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
        self.train_step_per_epoch = train_step_per_epoch
        self.test_step_per_epoch = test_step_per_epoch
        self.valid_step_per_epoch = valid_step_per_epoch

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
        num_channels = image.get_shape().as_list()[-1]
        channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
        for i in range(num_channels):
            channels[i] -= means[i]
        return tf.concat(axis=2, values=channels)

    def _train_parse_func(self, filename, label):
        image_string = tf.read_file(filename)
        image = tf.image.decode_jpeg(image_string, channels=3)
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

    def glimpse_network(self, img, loc, init):
        def retina_sensor(img_ph, loc, init=False):
            ll = tf.stop_gradient(loc)
            if init:
                pth = img_ph
            else:
                pth = tf.image.extract_glimpse(img_ph, [self.pth_size, self.pth_size], ll)
                # pth = tf.image.pad_to_bounding_box(pth, int((224-self.pth_size)/2), int((224-self.pth_size)/2), 224, 224)
                pth = tf.image.resize_images(pth, [224, 224])
            return pth

        def feature_extractor(patch):
            with slim.arg_scope(resnet_v1.resnet_arg_scope()):
                net, end_points = resnet_v1.resnet_v1_50(patch, 1000, is_training=self.train_mode, reuse=tf.AUTO_REUSE)
            resnet_feature = end_points['resnet_v1_50/block4']
            resnet_feature = tf.reduce_mean(resnet_feature, [1, 2], keepdims=True)
            resnet_feature = tf.squeeze(resnet_feature)
            resnet_feature = tf.reshape(resnet_feature, [-1, 2048])
            drop = tf.layers.dropout(resnet_feature, rate=0.2, training=self.train_mode)
            glimpse_feature = tf.layers.dense(inputs=drop, units=512,
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
        l = loc_feature_extractor(loc)
        return rgb, tf.nn.relu(tf.concat([l, g], axis=-1))
        # return tf.nn.relu(l+g)

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
        baseline = tf.transpose(tf.squeeze(baseline), (1, 0))
        return baseline

    def prob_network(self, rnn_last_out):
        logits = tf.layers.dense(rnn_last_out, 200, kernel_initializer=tf.glorot_uniform_initializer(),
                                 bias_initializer=tf.constant_initializer(0.1),
                                 # kernel_regularizer=tf.nn.l2_loss,
                                 name='prob/fc', reuse=tf.AUTO_REUSE)
        probs = tf.nn.softmax(logits)
        return logits, probs

    def loop_function(self, prev, step):
        loc, loc_mean = self.location_network(prev, True)
        if step > 0:
            rgb, glimpse = self.glimpse_network(self.img_ph, loc, False)
        else:
            rgb, glimpse = self.glimpse_network(self.img_ph, loc, True)
        return rgb, glimpse, loc, loc_mean

    def rnn_decode(self, initial_state, cell):
        state = initial_state
        outputs, locs, loc_means = [], [], []
        rgbs = []
        prev = tf.zeros([tf.shape(self.img_ph)[0], 256])
        for i in range(self.num_glimpses):
            rgb, inp, loc, loc_mean = self.loop_function(prev, i)
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
        self.rgbs, self.rnn_outputs, self.locs, self.loc_means = self.rnn_decode(init_state, cell)

        baselines = self.baseline_network(self.rnn_outputs)
        logits, probs = self.prob_network(self.rnn_outputs[-1])
        predict_label = tf.argmax(logits, 1, output_type=tf.int32)

        rewards = tf.cast(tf.equal(predict_label, self.lbl_ph), tf.float32)
        rewards = tf.tile(tf.expand_dims(rewards, 1), [1, self.num_glimpses])
        advantages = tf.stop_gradient(rewards - baselines)
        log_action_prob = self._log_likelihood(self.loc_means, self.locs, self.variance)

        self.classification_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.lbl_ph, logits=logits))
        self.reinforce_loss = tf.reduce_mean(log_action_prob * advantages)
        self.baselines_loss = tf.reduce_mean(tf.square((tf.stop_gradient(rewards) - baselines)))
        self.location_loss = self._loc_loss(self.loc_means)
        self.loss = -self.reinforce_loss + self.classification_loss + self.baselines_loss

        # without baseline
        # self.classification_loss = tf.reduce_mean(
        #     tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.lbl_ph, logits=logits))
        # self.reinforce_loss = tf.reduce_mean(log_action_prob * rewards)
        # self.location_loss = self._loc_loss(self.loc_means)
        # self.loss = -self.reinforce_loss + self.classification_loss + self.location_loss

        params = [param for param in tf.trainable_variables() if 'resnet' not in param.name]
        gradients = tf.gradients(self.loss, params)
        clipped_gradients, norm = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
        self.train_op = tf.train.AdamOptimizer(self.decay_learning_rate).apply_gradients(
            zip(clipped_gradients, params), global_step=self.global_step)

        self.test_label = tf.argmax(tf.reduce_mean(tf.reshape(probs, [self.glimpse_times, -1, 200]), axis=0), 1, output_type=tf.int32)
        self.test_acc = tf.reduce_mean(tf.cast(tf.equal(predict_label, self.labels), tf.float32))

        self.reward = tf.reduce_mean(rewards)
        self.advantage = tf.reduce_mean(advantages)

        tf.summary.histogram('probs', tf.reduce_mean(probs * tf.one_hot(self.lbl_ph, 200), axis=-1))
        for param in params:
            tf.summary.histogram(param.name, param)
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('reward', self.reward)
        tf.summary.scalar('advantage', self.advantage)
        tf.summary.scalar('classification_loss', self.classification_loss)
        tf.summary.scalar('reinforce_loss', self.reinforce_loss)
        tf.summary.scalar('baseline_loss', self.baselines_loss)
        self.merged = tf.summary.merge_all()

    def run(self):
        self.sess.run(tf.global_variables_initializer())
        init_fn = slim.assign_from_checkpoint_fn('resnet_v1_50.ckpt', slim.get_model_variables('resnet_v1_50'))
        init_fn(self.sess)

        saver = tf.train.Saver()
        cur_time = str(time.time())
        train_writer = tf.summary.FileWriter('log/ram/' + str(self.learning_rate) + '_' + cur_time + '/board/train', self.sess.graph)
        test_writer = tf.summary.FileWriter('log/ram/' + str(self.learning_rate) + '_' + cur_time + '/board/test', self.sess.graph)
        valid_writer = tf.summary.FileWriter('log/ram/' + str(self.learning_rate) + '_' + cur_time + '/board/valid', self.sess.graph)

        logging.info(str(self.learning_rate))
        logging.info(cur_time)

        for epoch in range(self.train_epoch):
            self.sess.run(self.train_init_op)
            for train_step in range(self.train_step_per_epoch):
                # origin, imgs, rgbs = self.sess.run([self.images, self.img_ph, self.rgbs], feed_dict={self.train_mode: True})
                # Image.fromarray(np.uint8(origin[0])).show('origin')
                # Image.fromarray(np.uint8(imgs[0])).show('tile')
                # for i, rgb in enumerate(rgbs):
                #     Image.fromarray(np.uint8(rgb[0])).show('patch' + str(i))

                _, summary, loss, reward, advantage, cls_loss, rein_loss, base_loss = self.sess.run(
                    [self.train_op, self.merged, self.loss, self.reward, self.advantage, self.classification_loss,
                     self.reinforce_loss, self.baselines_loss],
                    feed_dict={self.train_mode: True})

                train_writer.add_summary(summary, epoch * self.train_step_per_epoch + train_step)

                if train_step and train_step % self.logging_step == 0:
                    logging.info('Epoch {} Train step {}: loss = {:3.4f}\t reward = {:3.4f} \t advantage = {:3.4f} \t '
                                 'cls_loss = {:3.4f} \t rein_loss = {:3.4f} \t base_loss = {:3.4f}'
                                 .format(epoch, train_step, loss, reward, advantage, cls_loss, rein_loss, base_loss))

            saver.save(self.sess, 'log/ram/' + str(self.learning_rate) + '_' + cur_time + '/tmp/model.ckpt',
                       global_step=epoch)

            self.sess.run(self.valid_init_op)
            statistics = []
            for valid_step in range(self.valid_step_per_epoch):
                summary, loss, acc, reward, advantage, cls_loss, rein_loss, base_loss = self.sess.run(
                    [self.merged, self.loss, self.test_acc, self.reward, self.advantage, self.classification_loss,
                     self.reinforce_loss, self.baselines_loss],
                    feed_dict={self.train_mode: False})
                valid_writer.add_summary(summary, epoch * self.valid_step_per_epoch + valid_step)
                statistics.append([loss, acc, reward, advantage, cls_loss, rein_loss, base_loss])
            statistics = np.mean(np.asarray(statistics, np.float32), axis=0)
            logging.info('Epoch {} Valid: loss = {:3.4f}\t acc = {:3.4f} \t reward = {:3.4f} \t advantage = {:3.4f} \t '
                         'cls_loss = {:3.4f} \t rein_loss = {:3.4f} \t base_loss = {:3.4f}'.format(epoch, *statistics))

            if epoch and epoch % 5 == 0:
                self.sess.run(self.test_init_op)
                statistics = []
                for test_step in range(self.test_step_per_epoch):
                    summary, loss, acc, reward, advantage, cls_loss, rein_loss, base_loss = self.sess.run(
                        [self.merged, self.loss, self.test_acc, self.reward, self.advantage, self.classification_loss,
                         self.reinforce_loss, self.baselines_loss],
                        feed_dict={self.train_mode: False})
                    test_writer.add_summary(summary, epoch * self.test_step_per_epoch + test_step)
                    statistics.append([loss, acc, reward, advantage, cls_loss, rein_loss, base_loss])
                statistics = np.mean(np.asarray(statistics, np.float32), axis=0)
                logging.info('Epoch {} Test: loss = {:3.4f}\t acc = {:3.4f} \t reward = {:3.4f} \t advantage = {:3.4f} \t '
                             'cls_loss = {:3.4f} \t rein_loss = {:3.4f} \t base_loss = {:3.4f}'.format(epoch, *statistics))

        train_writer.close()
        test_writer.close()
        valid_writer.close()


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    with tf.Session() as sess:
        model = RecurrentAttentionModel(sess, **config.recurrent_attention_model)
        model.run()
