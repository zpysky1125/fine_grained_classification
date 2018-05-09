import tensorlayer as tl
import tensorflow as tf
import os
import numpy as np
import vgg19_trainable as vgg19
import utils


batch_size = 32

path = os.getcwd()


def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image_resized = tf.image.resize_images(image_decoded, [224, 224])
    image = tf.cast(image_resized, tf.float32)
    return image, label


train_test_split = {}
train_image_names, train_labels = [], []
valid_image_names, valid_labels = [], []
test_image_names, test_labels = [], []

with open(path + '/CUB_200_2011/CUB_200_2011/train_test_split.txt') as f:
    for l in f.readlines():
        l = l.strip('\n').split(' ')
        train_test_split[l[0]] = int(l[1])

with open(path + '/CUB_200_2011/CUB_200_2011/images.txt', 'r') as f:
    for l in f.readlines():
        l = l.strip('\n').split(' ')
        if train_test_split[l[0]] == 0:
            test_image_names.append(path + '/CUB_200_2011/CUB_200_2011/images/' + l[1])
        else:
            train_image_names.append(path + '/CUB_200_2011/CUB_200_2011/images/' + l[1])

with open(path + '/CUB_200_2011/CUB_200_2011/image_class_labels.txt', 'r') as f:
    for l in f.readlines():
        l = l.strip('\n').split(' ')
        if train_test_split[l[0]] == 0:
            test_labels.append(int(l[1]))
        else:
            train_labels.append(int(l[1]))



dataset = tf.data.Dataset.from_tensor_slices((train_image_names, train_labels))
dataset = dataset.map(_parse_function)
# dataset = dataset.shuffle(buffer_size=10000)
dataset = dataset.batch(batch_size)
iterator = dataset.make_one_shot_iterator()

next_images, next_labels = iterator.get_next()


# with tf.Session() as sess:
#     images = tf.placeholder(tf.float32, [batch_size, 224, 224, 3])
#     labels = tf.placeholder(tf.int32, [batch_size])
#     train_mode = tf.placeholder(tf.bool)
#
#     vgg = vgg19.Vgg19('./vgg19.npy')
#     vgg.build(images, train_mode)
#
#     print "Initialize Variables"
#     sess.run(tf.global_variables_initializer())
#
#     loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=vgg.fc8))
#     train = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)
#
#     for _ in range(100):
#         batch_images, batch_labels = sess.run([next_images, next_labels])
#         _, ll = sess.run([train, loss], feed_dict={images: batch_images, labels: batch_labels, train_mode: True})
#         print ll


    # for _ in range(100):
    #     batch_images, batch_labels = sess.run([next_images, next_labels])
    #     images = tf.placeholder("float", [32, 224, 224, 3])
    #     feed_dict = {images: batch_images}
    #
    #     prob = sess.run(vgg.prob, feed_dict=feed_dict)
    #     print(prob)
    #     for p in prob:
    #         utils.print_prob(p, './synset.txt'),
