import tensorflow as tf
import vgg19_trainable as vgg19
import time
from tfRecord import read_and_decode

batch_size = 16
valid_batch_size = 16
valid_loss = 0.0
valid_corrent_num = 0
valid_num = 900
train_image_num = 5094
valid_image_num = 900

images = tf.placeholder(tf.float32, [batch_size, 224, 224, 3])
labels = tf.placeholder(tf.int32, [batch_size])
train_mode = tf.placeholder(tf.bool)

vgg = vgg19.Vgg19('./vgg19.npy')
vgg.build(images, train_mode)

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=vgg.fc8))
train = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)

correct_prediction = tf.equal(tf.cast(tf.argmax(vgg.fc8, 1), tf.int32), labels)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
num_correct_preds = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

start = time.time()


for epoch in range(100):

    train_img, train_label = read_and_decode("train.tfrecords")
    print train_img, train_label
    next_images, next_labels = tf.train.shuffle_batch([train_img, train_label], batch_size=batch_size, capacity=10000+3*batch_size,
                                                      min_after_dequeue=10000)
    # next_images, next_labels = tf.train.batch([train_img, train_label], batch_size=batch_size)
    # valid_img, valid_label = read_and_decode("valid.tfrecords")
    # next_valid_img, next_valid_label = tf.train.shuffle_batch([valid_img, valid_label], batch_size=valid_batch_size,
    #                                                           capacity=30000,
    #                                                           min_after_dequeue=900)

    with tf.Session() as sess:
        print "Initialize Variables"
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        ind = 0
        try:
            while not coord.should_stop():
                # Run training steps or whatever
                batch_images, batch_labels = sess.run([next_images, next_labels])
                print batch_images.shape, batch_labels.shape
                print ind
                ind += 1
                print batch_labels

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()



        # train_batch = int(train_image_num / batch_size)
        # valid_batch = int(valid_image_num / valid_batch_size)
        # for i in range(train_batch):
        #     batch_images, batch_labels = sess.run([next_images, next_labels])
        #     print batch_images.shape, batch_labels.shape
        #     print i
        #     print batch_labels
            # _, batch_loss = sess.run([train, loss],
            #                          feed_dict={images: batch_images, labels: batch_labels, train_mode: True})
            # if i % 20 == 0:
            #     print ("VGG fine-tuning for {} batches in {} seconds".format(i, time.time() - start))
            #     print ("Epoch: {} step: {} loss: {}".format(epoch + 1, i, batch_loss))
            #     print ("Training Accuracy: {}".format(
            #         accuracy.eval(feed_dict={images: batch_images, labels: batch_labels, train_mode: True})))

        # valid_loss = 0.0
        # valid_corrent_num = 0
        # for i in range(valid_batch):
        #     batch_images, batch_labels = sess.run([next_valid_img, next_valid_label])
        #     valid_batch_correct_num, valid_batch_loss = sess.run([num_correct_preds, loss],
        #                                                          feed_dict={images: batch_images, labels: batch_labels,
        #                                                                     train_mode: True})
        #     valid_loss += valid_batch_loss
        #     valid_corrent_num += valid_batch_correct_num
        # print("Epoch: {}".format(epoch))
        # print("Validation Loss: {}".format(valid_loss))
        # print("Correct_val_count: {}  Total_val_count: {}".format(valid_corrent_num, valid_image_num))
        # print("Validation Data Accuracy: {}".format(
        #     100.0 * valid_corrent_num / (1.0 * valid_image_num)))  # for _ in range(100):
        coord.request_stop()
        coord.join(threads)
