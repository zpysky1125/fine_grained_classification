import tensorflow as tf
import vgg19_trainable as vgg19
import time
from tfRecord import read_and_decode
from tfRecord import get_batch

train_batch_size = 64
valid_batch_size = 64
valid_loss = 0.0
valid_corrent_num = 0
valid_num = 900
train_image_num = 5094
valid_image_num = 900
test_image_num = 5794
test_batch_size = 5794
train_epoches = 20
valid_epoches = 40

images = tf.placeholder(tf.float32, [None, 224, 224, 3])
labels = tf.placeholder(tf.int64, [None])
train_mode = tf.placeholder(tf.bool)

vgg = vgg19.Vgg19('./vgg19.npy')
vgg.build(images, train_mode)

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=vgg.fc8))
train = tf.train.AdagradOptimizer(0.001).minimize(loss)

correct_prediction = tf.equal(tf.argmax(vgg.fc8, 1), labels)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
num_correct_preds = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

start = time.time()

train_image_batch, train_label_batch = get_batch("train.tfrecords", train_batch_size)
valid_image_batch, valid_label_batch = get_batch("valid.tfrecords", valid_batch_size)
test_image_batch, test_label_batch = get_batch("test.tfrecords", test_batch_size)

# valid_img, valid_label = [1]*valid_epoches, [1]*valid_epoches
# next_valid_img, next_valid_label = [1]*valid_epoches, [1]*valid_epoches
#
#
# train_img, train_label = read_and_decode("train.tfrecords")
# next_images, next_labels = tf.train.shuffle_batch([train_img, train_label], batch_size=batch_size, capacity=6000+3*batch_size,
#                                                   min_after_dequeue=6000)
#
# valid_img, valid_label = read_and_decode("valid.tfrecords")
# valid_img_batch, valid_label_batch = tf.train.batch([valid_img, valid_label], batch_size=valid_batch_size, capacity=5000)
# next_valid_img, next_valid_label = tf.train.shuffle_batch([valid_img, valid_label], batch_size=valid_batch_size,
#                                                           capacity=250+3*valid_batch_size,
#                                                           min_after_dequeue=250)
# next_valid_img, next_valid_label = tf.train.batch([valid_img, valid_label], batch_size=valid_batch_size,
#                                                           capacity=900)

# with tf.Session() as sess:
#     print "Initialize Variables"
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#     sess.run(tf.local_variables_initializer())
#     sess.run(tf.global_variables_initializer())
#     try:
#         train_batch = int(train_image_num / batch_size)
#         valid_batch = int(valid_image_num / valid_batch_size)
#         for i in range(train_batch * 20):
#             batch_images, batch_labels = sess.run([train_image_batch, train_label_batch])
#             _, batch_loss = sess.run([train, loss],
#                                      feed_dict={images: batch_images, labels: batch_labels, train_mode: True})
#             if i % 20 == 0:
#                 print ("VGG fine-tuning for {} batches in {} seconds".format(i, time.time() - start))
#                 print ("step: {} loss: {}".format(i, batch_loss))
#                 print ("Training Accuracy: {}".format(
#                     accuracy.eval(feed_dict={images: batch_images, labels: batch_labels, train_mode: True})))
#
#             if i % 200 == 0:
#                 j = i/200
#                 valid_loss = 0.0
#                 valid_corrent_num = 0
#                 for i in range(valid_batch+1):
#                     batch_images, batch_labels = sess.run([next_valid_img[j], next_valid_label[j]])
#                     print batch_labels
#                     valid_batch_correct_num, valid_batch_loss = sess.run([num_correct_preds, loss],
#                                                                          feed_dict={images: batch_images,
#                                                                                     labels: batch_labels,
#                                                                                     train_mode: True})
#                     valid_loss += valid_batch_loss
#                     valid_corrent_num += valid_batch_correct_num
#                 print("Validation Loss: {}".format(valid_loss))
#                 print("Correct_val_count: {}  Total_val_count: {}".format(valid_corrent_num, valid_image_num))
#                 print("Validation Data Accuracy: {}".format(
#                     100.0 * valid_corrent_num / (1.0 * valid_image_num)))  # for _ in range(100):
#
#     except tf.errors.OutOfRangeError:
#         print('Done training -- epoch limit reached')
#     finally:
#         coord.request_stop()
#
#     # coord.request_stop()
#     coord.join(threads)



with tf.Session() as sess:
    print "Initialize Variables"
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    try:
        train_batch = int(train_image_num / train_batch_size)
        valid_batch = int(valid_image_num / valid_batch_size)
        for i in range(50):

            if coord.should_stop():
                break
            for j in range(train_batch + 1):
                train_image, train_label = sess.run([train_image_batch, train_label_batch])
                _, batch_loss = sess.run([train, loss],
                                         feed_dict={images: train_image, labels: train_label, train_mode: True})
                print ("VGG fine-tuning for {} batches in {} seconds".format(j, time.time() - start))
                print ("step: {} loss: {}".format(j, batch_loss))
                print ("Training Accuracy: {}".format(
                    accuracy.eval(feed_dict={images: train_image, labels: train_label, train_mode: True})))

            valid_loss = 0.0
            valid_corrent_num = 0

            for j in range(valid_batch + 1):
                valid_image, valid_label = sess.run([valid_image_batch, valid_label_batch])
                valid_batch_corrent_num, valid_batch_loss = sess.run([num_correct_preds, loss],
                                                                     feed_dict={images: valid_image,
                                                                                labels: valid_label,
                                                                                train_mode: True})
                valid_loss += valid_batch_loss
                valid_corrent_num += valid_batch_corrent_num

            print("Validation Loss: {}".format(valid_loss))
            print("Correct_val_count: {}  Total_val_count: {}".format(valid_corrent_num, valid_image_num))
            print("Validation Data Accuracy: {}".format(100.0 * valid_corrent_num / (1.0 * valid_image_num)))

            # if i % 20 == 0:
            #     test_image, test_label = sess.run([test_image_batch, test_label_batch])
            #
            #     test_corrent_num, test_loss = sess.run([num_correct_preds, loss],
            #                                            feed_dict={images: test_image,
            #                                                       labels: test_label,
            #                                                       train_mode: True})
            #     print("Test Loss: {}".format(test_loss))
            #     print("Correct_test_count: {}  Total_test_count: {}".format(test_corrent_num, test_image_num))
            #     print("Test Data Accuracy: {}".format(100.0 * test_corrent_num / (1.0 * test_image_num)))
    except tf.errors.OutOfRangeError:
        print('Done!')
    finally:
        coord.request_stop()
    coord.join(threads)
