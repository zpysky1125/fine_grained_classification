import tensorlayer as tl
import tensorflow as tf
import os
import numpy as np
import vgg19_trainable as vgg19
import utils
import time
from sklearn.model_selection import train_test_split

batch_size = 16

path = os.getcwd()


def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image_resized = tf.image.resize_images(image_decoded, [224, 224])
    image = tf.cast(image_resized, tf.float32)
    return image, label


dict_train_test_split = {}
train_valid_image_names, train_valid_labels = [], []
train_image_names, train_labels = [], []
valid_image_names, valid_labels = [], []
test_image_names, test_labels = [], []

with open(path + '/CUB_200_2011/CUB_200_2011/train_test_split.txt') as f:
    for l in f.readlines():
        l = l.strip('\n').split(' ')
        dict_train_test_split[l[0]] = int(l[1])

with open(path + '/CUB_200_2011/CUB_200_2011/images.txt', 'r') as f:
    for l in f.readlines():
        l = l.strip('\n').split(' ')
        if dict_train_test_split[l[0]] == 0:
            test_image_names.append(path + '/CUB_200_2011/CUB_200_2011/images/' + l[1])
        else:
            train_valid_image_names.append(path + '/CUB_200_2011/CUB_200_2011/images/' + l[1])

with open(path + '/CUB_200_2011/CUB_200_2011/image_class_labels.txt', 'r') as f:
    for l in f.readlines():
        l = l.strip('\n').split(' ')
        if dict_train_test_split[l[0]] == 0:
            test_labels.append(int(l[1]))
        else:
            train_valid_labels.append(int(l[1]))

with tf.device('/gpu:1'):
    with tf.Session() as sess:
        images = tf.placeholder(tf.float32, [batch_size, 224, 224, 3])
        labels = tf.placeholder(tf.int32, [batch_size])
        train_mode = tf.placeholder(tf.bool)

        vgg = vgg19.Vgg19('./vgg19.npy')
        vgg.build(images, train_mode)

        print "Initialize Variables"
        sess.run(tf.global_variables_initializer())

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=vgg.fc8))
        train = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)

        correct_prediction = tf.equal(tf.argmax(vgg.fc8, 1, output_type=tf.int32), labels)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        num_correct_preds = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

        start = time.time()

        for epoch in range(100):
            train_image_names, valid_image_names, train_labels, valid_labels = train_test_split(train_valid_image_names,
                                                                                                train_valid_labels,
                                                                                                test_size=0.15,
                                                                                                shuffle=True)
            dataset = tf.data.Dataset.from_tensor_slices((train_image_names, train_labels))
            dataset = dataset.map(_parse_function)
            dataset = dataset.batch(batch_size)
            iterator = dataset.make_one_shot_iterator()
            next_images, next_labels = iterator.get_next()

            avg_cost = 0.
            total_batch = int(6000 / batch_size)

            for i in range(total_batch):
                batch_images, batch_labels = sess.run([next_images, next_labels])
                _, batch_loss = sess.run([train, loss],
                                         feed_dict={images: batch_images, labels: batch_labels, train_mode: True})
                if i % 20 == 0:
                    print ("VGG fine-tuning for {} batches in % seconds: {}".format(i * 20, time.time() - start))
                    print ("Epoch: {} step: {} loss: {}".format(epoch + 1, i, batch_loss))
                    print ("Training Accuracy: {}".format(
                        accuracy.eval(feed_dict={images: batch_images, labels: batch_labels, train_mode: True})))

            valid_batch_size = 16
            valid_dataset = tf.data.Dataset.from_tensor_slices((valid_image_names, valid_labels))
            valid_dataset = valid_dataset.map(_parse_function)
            valid_dataset = valid_dataset.batch(batch_size)
            iterator = valid_dataset.make_one_shot_iterator()
            next_images, next_labels = iterator.get_next()
            valid_loss = 0.0
            valid_corrent_num = 0

            for i in range(len(valid_labels) / valid_batch_size):
                batch_images, batch_labels = sess.run([next_images, next_labels])
                valid_batch_correct_num, valid_batch_loss = sess.run([num_correct_preds, loss],
                                                                     feed_dict={images: batch_images,
                                                                                labels: batch_labels, train_mode: True})
                valid_loss += valid_batch_loss
                valid_corrent_num += valid_batch_correct_num

            print("Epoch: {}".format(epoch))
            print("Validation Loss: {}".format(valid_loss))
            print("Correct_val_count: {}  Total_val_count: {}".format(valid_corrent_num, len(valid_labels)))
            print("Validation Data Accuracy: {}".format(100.0 * valid_corrent_num / (1.0 * len(valid_labels))))


# for _ in range(100):
#     batch_images, batch_labels = sess.run([next_images, next_labels])
#     images = tf.placeholder("float", [32, 224, 224, 3])
#     feed_dict = {images: batch_images}
#
#     prob = sess.run(vgg.prob, feed_dict=feed_dict)
#     print(prob)
#     for p in prob:
#         utils.print_prob(p, './synset.txt'),
