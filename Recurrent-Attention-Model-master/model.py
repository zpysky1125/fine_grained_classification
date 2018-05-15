import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import BasicLSTMCell
from tensorflow.contrib.legacy_seq2seq.python.ops.seq2seq import rnn_decoder
from tensorflow.python.ops.distributions.normal import Normal
import numpy as np
import os
import inspect


def _weight_variable(shape):
    initial = tf.truncated_normal(shape=shape, stddev=.01)
    return tf.Variable(initial)


def _bias_variable(shape):
    # initial = tf.truncated_normal(shape=shape, stddev=.001)
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)


def _log_likelihood(loc_means, locs, variance):
    loc_means = tf.stack(loc_means)  # [timesteps, batch_sz, loc_dim]
    locs = tf.stack(locs)
    gaussian = Normal(loc_means, variance)
    logll = gaussian._log_prob(x=locs)  # [timesteps, batch_sz, loc_dim]
    logll = tf.reduce_sum(logll, 2)
    return tf.transpose(logll)  # [batch_sz, timesteps]


class RetinaSensor(object):
    # one scale
    def __init__(self, img_size, pth_size):
        self.img_size = img_size
        self.pth_size = pth_size

    def __call__(self, img_ph, loc):
        pth = tf.image.extract_glimpse(img_ph, [self.pth_size, self.pth_size], loc)
        # pth = tf.image.resize_images(pth, [224, 224])
        return pth


class GlimpseNetwork(object):
    def __init__(self, img_size, pth_size, loc_dim, g_size, l_size, output_size, vgg16_npy_path=None):
        self.retina_sensor = RetinaSensor(img_size, pth_size)
        self.g_size = g_size
        self.l_size = l_size
        self.output_size = output_size
        self.loc_dim = loc_dim

        # location network weight

        # # layer 1
        # self.g1_w = _weight_variable((pth_size * pth_size, g_size))
        # self.g1_b = _bias_variable((g_size,))
        # self.g2_b = _bias_variable((output_size,))
        # self.l2_w = _weight_variable((l_size, output_size))
        # self.l2_b = _bias_variable((output_size,))
        self.l1_w = _weight_variable((self.loc_dim, self.l_size))
        self.l1_b = _bias_variable((self.l_size,))
        self.l2_w = _weight_variable((self.l_size, self.output_size))
        self.l2_b = _bias_variable((self.output_size,))

        # path network weight

        if vgg16_npy_path is None:
            path = inspect.getfile(GlimpseNetwork)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, "vgg16.npy")
            vgg16_npy_path = path
            print(path)

        # self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        self.data_dict = None

        print("npy file loaded")

        self.conv1_1_weight, self.conv1_1_bias = self.get_conv_var(3, 3, 64, "conv1_1")
        self.conv1_2_weight, self.conv1_2_bias = self.get_conv_var(3, 64, 64, "conv1_2")
        self.conv2_1_weight, self.conv2_1_bias = self.get_conv_var(3, 64, 128, "conv2_1")
        self.conv2_2_weight, self.conv2_2_bias = self.get_conv_var(3, 128, 128, "conv2_2")
        self.conv3_1_weight, self.conv3_1_bias = self.get_conv_var(3, 128, 256, "conv3_1")
        self.conv3_2_weight, self.conv3_2_bias = self.get_conv_var(3, 256, 256, "conv3_2")
        self.conv3_3_weight, self.conv3_3_bias = self.get_conv_var(3, 256, 256, "conv3_3")
        self.conv4_1_weight, self.conv4_1_bias = self.get_conv_var(3, 256, 512, "conv4_1")
        self.conv4_2_weight, self.conv4_2_bias = self.get_conv_var(3, 512, 512, "conv4_2")
        self.conv4_3_weight, self.conv4_3_bias = self.get_conv_var(3, 512, 512, "conv4_3")
        self.conv5_1_weight, self.conv5_1_bias = self.get_conv_var(3, 512, 512, "conv5_1")
        self.conv5_2_weight, self.conv5_2_bias = self.get_conv_var(3, 512, 512, "conv5_2")
        self.conv5_3_weight, self.conv5_3_bias = self.get_conv_var(3, 512, 512, "conv5_3")

        self.VGG_MEAN = [103.939, 116.779, 123.68]
        self.dropout = 0.5

    def __call__(self, imgs_ph, locs, train_mode=None):
        rgb = self.retina_sensor(imgs_ph, locs)
        g = self.patch_feature_extractor(rgb, train_mode)
        l = self.loc_feature_extractor(locs)
        print g.get_shape()
        print l.get_shape()
        return tf.nn.relu(l + g)

        # g = tf.nn.xw_plus_b(tf.nn.relu(tf.nn.xw_plus_b(pths, self.g1_w, self.g1_b)), self.g2_w, self.g2_b)
        # l = tf.nn.xw_plus_b(tf.nn.relu(tf.nn.xw_plus_b(locs, self.l1_w, self.l1_b)), self.l2_w, self.l2_b)
        # return tf.nn.relu(g + l)

    def loc_feature_extractor(self, locs):
        # self.l1_w = _weight_variable((self.loc_dim, self.l_size))
        # self.l1_b = _bias_variable((self.l_size,))
        # self.l2_w = _weight_variable((self.l_size, self.output_size))
        # self.l2_b = _bias_variable((self.output_size,))
        l = tf.nn.xw_plus_b(tf.nn.relu(tf.nn.xw_plus_b(locs, self.l1_w, self.l1_b)), self.l2_w, self.l2_b)
        return l

    # vgg 16 network
    def patch_feature_extractor(self, rgb, train_mode=None):
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb)
        # assert red.get_shape().as_list()[1:] == [224, 224, 1]
        # assert green.get_shape().as_list()[1:] == [224, 224, 1]
        # assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(axis=3, values=[
            blue - self.VGG_MEAN[0],
            green - self.VGG_MEAN[1],
            red - self.VGG_MEAN[2],
        ])
        # assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        self.conv1_1 = self.conv_layer(bgr, 3, 64, "conv1_1", self.conv1_1_weight, self.conv1_1_bias)
        self.conv1_2 = self.conv_layer(self.conv1_1, 64, 64, "conv1_2", self.conv1_2_weight, self.conv1_2_bias)

        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, 64, 128, "conv2_1", self.conv2_1_weight, self.conv2_1_bias)
        self.conv2_2 = self.conv_layer(self.conv2_1, 128, 128, "conv2_2", self.conv2_2_weight, self.conv2_2_bias)
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, 128, 256, "conv3_1", self.conv3_1_weight, self.conv3_1_bias)
        self.conv3_2 = self.conv_layer(self.conv3_1, 256, 256, "conv3_2", self.conv3_2_weight, self.conv3_2_bias)
        self.conv3_3 = self.conv_layer(self.conv3_2, 256, 256, "conv3_3", self.conv3_3_weight, self.conv3_3_bias)
        self.pool3 = self.max_pool(self.conv3_3, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, 256, 512, "conv4_1", self.conv4_1_weight, self.conv4_1_bias)
        self.conv4_2 = self.conv_layer(self.conv4_1, 512, 512, "conv4_2", self.conv4_2_weight, self.conv4_2_bias)
        self.conv4_3 = self.conv_layer(self.conv4_2, 512, 512, "conv4_3", self.conv4_3_weight, self.conv4_3_bias)
        self.pool4 = self.max_pool(self.conv4_3, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, 512, 512, "conv5_1", self.conv5_1_weight, self.conv5_1_bias)
        self.conv5_2 = self.conv_layer(self.conv5_1, 512, 512, "conv5_2", self.conv5_2_weight, self.conv5_2_bias)
        self.conv5_3 = self.conv_layer(self.conv5_2, 512, 512, "conv5_3", self.conv5_3_weight, self.conv5_3_bias)
        self.pool5 = self.avg_pool(self.conv5_3, "avg_pool_5")
        self.g = tf.reshape(self.pool5, [-1, 512])

        return self.g

        # self.conv1_1 = self.conv_layer(bgr, 3, 64, "conv1_1")
        # self.conv1_2 = self.conv_layer(self.conv1_1, 64, 64, "conv1_2")
        # self.pool1 = self.max_pool(self.conv1_2, 'pool1')
        #
        # self.conv2_1 = self.conv_layer(self.pool1, 64, 128, "conv2_1")
        # self.conv2_2 = self.conv_layer(self.conv2_1, 128, 128, "conv2_2")
        # self.pool2 = self.max_pool(self.conv2_2, 'pool2')
        #
        # self.conv3_1 = self.conv_layer(self.pool2, 128, 256, "conv3_1")
        # self.conv3_2 = self.conv_layer(self.conv3_1, 256, 256, "conv3_2")
        # self.conv3_3 = self.conv_layer(self.conv3_2, 256, 256, "conv3_3")
        # self.pool3 = self.max_pool(self.conv3_3, 'pool3')
        #
        # self.conv4_1 = self.conv_layer(self.pool3, 256, 512, "conv4_1")
        # self.conv4_2 = self.conv_layer(self.conv4_1, 512, 512, "conv4_2")
        # self.conv4_3 = self.conv_layer(self.conv4_2, 512, 512, "conv4_3")
        # self.pool4 = self.max_pool(self.conv4_3, 'pool4')
        #
        # self.conv5_1 = self.conv_layer(self.pool4, 512, 1024, "conv5_1")
        # self.conv5_2 = self.conv_layer(self.conv5_1, 1024, 1024, "conv5_2")
        # self.conv5_3 = self.conv_layer(self.conv5_2, 1024, 1024, "conv5_3")
        # # self.pool5 = self.max_pool(self.conv5_3, 'pool5')
        #
        # self.fc6 = self.avg_pool(self.conv5_3, "fc6")
        # self.fc6 = tf.reshape(self.fc6, [-1, 1024])

        # self.fc6 = self.fc_layer(self.pool3, 25088, 4096, "fc6")  # 25088 = ((224 // (2 ** 5)) ** 2) * 512
        # self.relu6 = tf.nn.relu(self.fc6)
        # if train_mode is not None:
        #     self.relu6 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu6, self.dropout), lambda: self.relu6)
        # else:
        #     self.relu6 = tf.nn.dropout(self.relu6, self.dropout)
        #
        # self.fc7 = self.fc_layer(self.relu6, 4096, 4096, "fc7")
        # self.relu7 = tf.nn.relu(self.fc7)
        # if train_mode is not None:
        #     self.relu7 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu7, self.dropout), lambda: self.relu7)
        # else:
        #     self.relu7 = tf.nn.dropout(self.relu7, self.dropout)

        # self.fc8 = self.fc_layer(self.relu7, 4096, 200, "fc8_fine")
        # self.prob = tf.nn.softmax(self.fc8, name="prob")

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 7, 7, 1], strides=[1, 7, 7, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    # def conv_layer(self, bottom, in_channels, out_channels, name):
        # with tf.variable_scope(name):
        # filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, name)
        # conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
        # bias = tf.nn.bias_add(conv, conv_biases)
        # relu = tf.nn.relu(bias)
        # return relu

    def conv_layer(self, bottom, in_channels, out_channels, name, filt, conv_biases):
        with tf.variable_scope(name):
            # filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, name)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)
            return relu

    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)
            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
            return fc

    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
        filters = self.get_var(initial_value, name, 0, name + "_filters")
        initial_value = tf.truncated_normal([out_channels], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")
        return filters, biases

    def get_fc_var(self, in_size, out_size, name):
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
        weights = self.get_var(initial_value, name, 0, name + "_weights")
        initial_value = tf.truncated_normal([out_size], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return weights, biases

    def get_var(self, initial_value, name, idx, var_name):
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
            print name + " get from dict"
        else:
            value = initial_value
            print name + " get from initial"
        return tf.Variable(value, name=var_name)


class LocationNetwork(object):
    def __init__(self, loc_dim, rnn_output_size, variance=0.22, is_sampling=False):
        self.loc_dim = loc_dim
        self.variance = variance
        self.w = _weight_variable((rnn_output_size, loc_dim))
        self.b = _bias_variable((loc_dim,))

        self.is_sampling = is_sampling

    def __call__(self, cell_output):
        mean = tf.nn.xw_plus_b(cell_output, self.w, self.b)
        mean = tf.clip_by_value(mean, -1., 1.)
        mean = tf.stop_gradient(mean)

        if self.is_sampling:
            loc = mean + tf.random_normal(
                (tf.shape(cell_output)[0], self.loc_dim),
                stddev=self.variance)
            loc = tf.clip_by_value(loc, -1., 1.)
        else:
            loc = mean
        loc = tf.stop_gradient(loc)
        return loc, mean


class RecurrentAttentionModel(object):
    def __init__(self, img_size, pth_size, g_size, l_size, glimpse_output_size,
                 loc_dim, variance,
                 cell_size, num_glimpses, num_classes,
                 learning_rate, learning_rate_decay_factor, min_learning_rate, training_steps_per_epoch,
                 max_gradient_norm,
                 is_training=False):

        self.img_ph = tf.placeholder(tf.float32, [None, img_size, img_size, 3])
        self.lbl_ph = tf.placeholder(tf.int64, [None])

        self.global_step = tf.Variable(0, trainable=False)

        self.learning_rate = learning_rate

        # self.learning_rate = tf.maximum(tf.train.exponential_decay(
        #     learning_rate, self.global_step,
        #     training_steps_per_epoch,
        #     learning_rate_decay_factor,
        #     staircase=True),
        #     min_learning_rate)

        cell = BasicLSTMCell(cell_size)

        with tf.variable_scope('GlimpseNetwork'):
            glimpse_network = GlimpseNetwork(img_size, pth_size, loc_dim, g_size, l_size, glimpse_output_size, './vgg16.npy')
        with tf.variable_scope('LocationNetwork'):
            location_network = LocationNetwork(loc_dim=loc_dim, rnn_output_size=cell.output_size, variance=variance,
                                               is_sampling=is_training)

        # Core Network
        batch_size = tf.shape(self.img_ph)[0]
        init_loc = tf.random_uniform((batch_size, loc_dim), minval=-1, maxval=1)
        init_state = cell.zero_state(batch_size, tf.float32)

        init_glimpse = glimpse_network(self.img_ph, init_loc)
        self.init_glip = init_glimpse
        rnn_inputs = [init_glimpse]
        rnn_inputs.extend([0] * num_glimpses)

        locs, loc_means = [], []

        def loop_function(prev, _):
            loc, loc_mean = location_network(prev)
            locs.append(loc)
            loc_means.append(loc_mean)
            glimpse = glimpse_network(self.img_ph, loc)
            return glimpse

        rnn_outputs, _ = rnn_decoder(rnn_inputs, init_state, cell, loop_function=loop_function)

        # Time independent baselines
        with tf.variable_scope('Baseline'):
            baseline_w = _weight_variable((cell.output_size, 1))
            baseline_b = _bias_variable((1,))
        baselines = []
        for output in rnn_outputs[1:]:
            baseline = tf.nn.xw_plus_b(output, baseline_w, baseline_b)
            baseline = tf.squeeze(baseline)
            baselines.append(baseline)
        baselines = tf.stack(baselines)  # [timesteps, batch_sz]
        baselines = tf.transpose(baselines)  # [batch_sz, timesteps]

        # Classification. Take the last step only.
        rnn_last_output = rnn_outputs[-1]
        self.rnn_last = rnn_last_output
        with tf.variable_scope('Classification'):
            logit_w = _weight_variable((cell.output_size, num_classes))
            logit_b = _bias_variable((num_classes,))
        logits = tf.nn.xw_plus_b(rnn_last_output, logit_w, logit_b)
        self.logits = logits
        self.prediction = tf.argmax(logits, 1)
        self.softmax = tf.nn.softmax(logits)

        if is_training:
            # classification loss
            self.xent = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.lbl_ph, logits=logits))
            # RL reward
            reward = tf.cast(tf.equal(self.prediction, self.lbl_ph), tf.float32)
            rewards = tf.expand_dims(reward, 1)  # [batch_sz, 1]
            rewards = tf.tile(rewards, (1, num_glimpses))  # [batch_sz, timesteps]
            advantages = rewards - tf.stop_gradient(baselines)
            self.advantage = tf.reduce_mean(advantages)
            logll = _log_likelihood(loc_means, locs, variance)
            logllratio = tf.reduce_mean(logll * advantages)
            self.reward = tf.reduce_mean(reward)
            # baseline loss
            self.baselines_mse = tf.reduce_mean(tf.square((rewards - baselines)))
            # hybrid loss
            self.loss = -logllratio + self.xent + self.baselines_mse
            params = tf.trainable_variables()
            gradients = tf.gradients(self.loss, params)
            clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(
                zip(clipped_gradients, params), global_step=self.global_step)

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=99999999)
