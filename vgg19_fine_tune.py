import time
import sys
import tensorflow as tf

from vgg_network import vgg19_trainable as vgg19
# from tfRecord import get_batch, get_shuffle_batch
from input_generator import get_batch, BirdClassificationGenerator

sys.path.append('../')

train_image_num = 5094
valid_image_num = 900
test_image_num = 5794
train_batch_size = 64
valid_batch_size = 64
test_batch_size = 64

train_batch = train_image_num // train_batch_size if train_image_num % train_batch_size == 0 else train_image_num // train_batch_size + 1
valid_batch = valid_image_num // valid_batch_size if valid_image_num % valid_batch_size == 0 else valid_image_num // valid_batch_size + 1
test_batch = test_image_num // test_batch_size if test_image_num % test_batch_size == 0 else test_image_num // test_batch_size + 1


train_losses, valid_losses, test_losses = [], [], []
train_accuracys, valid_accuracys, test_accuracys = [], [], []

bird_classification_generator = BirdClassificationGenerator("./CUB_200_2011/CUB_200_2011/")
train_generator = bird_classification_generator.train_generator(train_batch_size)
valid_generator = bird_classification_generator.valid_generator(valid_batch_size)
test_generator = bird_classification_generator.test_generator(test_batch_size)

images = tf.placeholder(tf.float32, [None, 224, 224, 3])
labels = tf.placeholder(tf.int64, [None])
train_mode = tf.placeholder(tf.bool)

vgg = vgg19.Vgg19('./vgg19.npy')
vgg.build(images, train_mode)

# l1_regularizer = tf.contrib.layers.l1_regularizer(
#    scale=0.005, scope=None
# )
#
# weights = tf.trainable_variables()
# regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, weights)

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=vgg.fc8))
train = tf.train.MomentumOptimizer(0.0001, 0.9).minimize(loss)

correct_prediction = tf.equal(tf.argmax(vgg.fc8, 1), labels)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
num_correct_preds = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

start = time.time()

# train_image_batch, train_label_batch = get_shuffle_batch("train.tfrecords", train_batch_size)
# test_train_image_batch, test_train_label_batch = get_batch("train.tfrecords", train_batch_size)
# valid_image_batch, valid_label_batch = get_batch("valid.tfrecords", valid_batch_size)
# test_image_batch, test_label_batch = get_batch("test.tfrecords", test_batch_size)

saver = tf.train.Saver()

with tf.Session() as sess:
    print "Initialize Variables"

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    try:

        for i in range(100):
            if coord.should_stop():
                break
            for j in range(train_batch):
                # train_image, train_label = sess.run([train_image_batch, train_label_batch])
                train_image, train_label = get_batch(train_generator)
                _, batch_loss, acc = sess.run([train, loss, accuracy],
                                              feed_dict={images: train_image, labels: train_label, train_mode: True})
                # print ("Epoch: {} step: {} loss: {} accuracy: {} time: {} seconds".format(i, j, batch_loss, acc * 100.0, time.time() - start))

            train_loss = 0.0
            train_correct_num = 0

            for j in range(train_batch):
                # test_train_image, test_train_label = sess.run([test_train_image_batch, test_train_label_batch])
                test_train_image, test_train_label = get_batch(train_generator)
                train_batch_correct_num, train_batch_loss = sess.run([num_correct_preds, loss],
                                                                     feed_dict={images: test_train_image,
                                                                                labels: test_train_label,
                                                                                train_mode: False})
                train_loss += train_batch_loss
                train_correct_num += train_batch_correct_num

            print (time.time() - start)
            print ("Epoch: {}".format(i))
            print ("Train Loss: {}".format(train_loss))
            print ("Correct_train_count: {}  Total_train_count: {}".format(train_correct_num, train_image_num))
            print ("Train Data Accuracy: {}".format(100.0 * train_correct_num / (1.0 * train_image_num)))
            print

            train_losses.append(train_loss), train_accuracys.append(
                100.0 * train_correct_num / (1.0 * train_image_num))

            valid_loss = 0.0
            valid_corrent_num = 0

            for j in range(valid_batch):
                # valid_image, valid_label = sess.run([valid_image_batch, valid_label_batch])
                valid_image, valid_label = get_batch(valid_generator)
                valid_batch_correct_num, valid_batch_loss = sess.run([num_correct_preds, loss],
                                                                     feed_dict={images: valid_image,
                                                                                labels: valid_label,
                                                                                train_mode: False})
                valid_loss += valid_batch_loss
                valid_corrent_num += valid_batch_correct_num

            print
            print ("Epoch: {}".format(i))
            print ("Validation Loss: {}".format(valid_loss))
            print ("Correct_val_count: {}  Total_val_count: {}".format(valid_corrent_num, valid_image_num))
            print ("Validation Data Accuracy: {}".format(100.0 * valid_corrent_num / (1.0 * valid_image_num)))
            print

            valid_losses.append(valid_loss), valid_accuracys.append(
                100.0 * valid_corrent_num / (1.0 * valid_image_num))

            if (i + 1) % 10 == 0:
                test_loss = 0.0
                test_correct_num = 0
                for j in range(test_batch):
                    # test_image, test_label = sess.run([test_image_batch, test_label_batch])
                    test_image, test_label = get_batch(test_generator)
                    test_batch_correct_num, test_batch_loss = sess.run([num_correct_preds, loss],
                                                                       feed_dict={images: test_image,
                                                                                  labels: test_label,
                                                                                  train_mode: False})
                    test_loss += test_batch_loss
                    test_correct_num += test_batch_correct_num

                print
                print ("Epoch: {}", i)
                print ("Test Loss: {}".format(test_loss))
                print ("Correct_test_count: {}  Total_test_count: {}".format(test_correct_num, test_image_num))
                print ("Test Data Accuracy: {}".format(100.0 * test_correct_num / (1.0 * test_image_num)))
                print

                test_losses.append(test_loss), test_accuracys.append(100.0 * test_correct_num / (1.0 * test_image_num))

                saver.save(sess, './tmp/model.ckpt', global_step=i + 1)

    except tf.errors.OutOfRangeError:
        print('Done!')
    finally:
        coord.request_stop()
    coord.join(threads)

    f = open("res", 'w+')
    print >> f, train_losses
    print >> f, train_accuracys
    print >> f, valid_losses
    print >> f, valid_accuracys
    print >> f, test_losses
    print >> f, test_accuracys
