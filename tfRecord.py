import os
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
import vgg19_trainable as vgg19
import time

path = os.getcwd()

dict_train_test_split = {}
train_valid_image_names, train_valid_labels = [], []
train_image_names, train_labels = [], []
valid_image_names, valid_labels = [], []
test_image_names, test_labels = [], []

# sess = tf.Session()
sess = None


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(image_buffer, label, height, width):
    """Build an Example proto for an example.
    Args:
      filename: string, path to an image file, e.g., '/path/to/example.JPG'
      image_buffer: string, JPEG encoding of RGB image
      label: integer, identifier for the ground truth for the network
      text: string, unique human-readable, e.g. 'dog'
      height: integer, image height in pixels
      width: integer, image width in pixels
    Returns:
      Example proto
    """
    colorspace = 'RGB'
    channels = 3
    image_format = 'JPEG'

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/colorspace': _bytes_feature(tf.compat.as_bytes(colorspace)),
        'image/channels': _int64_feature(channels),
        'image/class/label': _int64_feature(label),
        'image/format': _bytes_feature(tf.compat.as_bytes(image_format)),
        'image/encoded': _bytes_feature(tf.compat.as_bytes(image_buffer))}))
    return example


def _process_image(filename):
    """Process a single image file.
    Args:
      filename: string, path to an image file e.g., '/path/to/example.JPG'.
      coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
      image_buffer: string, JPEG encoding of RGB image.
      height: integer, image height in pixels.
      width: integer, image width in pixels.
    """
    # Read the image file.
    with tf.gfile.FastGFile(filename, 'rb') as f:
        image_data = f.read()

    image = sess.run(tf.image.decode_jpeg(image_data, channels=3))

    # Check that image converted to RGB
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 3

    return image_data, height, width


def generate_test_valid_train_set():
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
                test_labels.append(int(l[1]) - 1)
            else:
                train_valid_labels.append(int(l[1]) - 1)


def create_record(image_names, image_labels, out_name):
    writer = tf.python_io.TFRecordWriter(out_name)
    for index, name in enumerate(image_names):
        img_path = name
        img = Image.open(img_path)
        img = img.resize((224, 224))
        img_raw = img.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[image_labels[index]])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))
        # image_buffer, height, width = _process_image(img_path)
        # example = _convert_to_example(image_buffer, image_labels[index], height, width)
        writer.write(example.SerializeToString())
    writer.close()


def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # features = tf.parse_single_example(serialized_example, features = {
    #     "image/encoded": tf.FixedLenFeature([], tf.string),
    #     "image/height": tf.FixedLenFeature([], tf.int64),
    #     "image/width": tf.FixedLenFeature([], tf.int64),
    #     "image/class/label": tf.FixedLenFeature([], tf.int64),})
    # image_encoded = features["image/encoded"]
    # image_raw = tf.image.decode_jpeg(image_encoded, channels=3)
    # img = tf.image.resize_image_with_crop_or_pad(image_raw, 224, 224)
    # label = tf.cast(features["image/class/label"], tf.int64)

    features = tf.parse_single_example(serialized_example, features={
        'label': tf.FixedLenFeature([], tf.int64),
        'img_raw': tf.FixedLenFeature([], tf.string)
    })
    img = tf.image.decode_jpeg(features['img_raw'], tf.uint8)
    img = tf.cast(img, tf.float32) * (1. / 255.0) - 0.5
    label = tf.cast(features['label'], tf.int64)
    return img, label


def get_batch(data, batch_size):
    img, label = read_and_decode(data)
    img_batch, label_batch = tf.train.shuffle_batch(
        [img, label], batch_size=batch_size, capacity=6000, min_after_dequeue=1000, allow_smaller_final_batch=True)
    return img_batch, label_batch


def main(unused_argv):
    generate_test_valid_train_set()
    train_image_names, valid_image_names, train_labels, valid_labels = train_test_split(train_valid_image_names,
                                                                                        train_valid_labels,
                                                                                        test_size=0.15,
                                                                                        shuffle=True)
    create_record(train_image_names, train_labels, "train.tfrecords")
    create_record(valid_image_names, valid_labels, "valid.tfrecords")
    create_record(test_image_names, test_labels, "test.tfrecords")

    # img, label = read_and_decode("train.tfrecords")
    # img_batch, label_batch = tf.train.shuffle_batch([img, label],
    #                                                 batch_size=20, capacity=100,
    #                                                 min_after_dequeue=5)
    # sess.run(tf.local_variables_initializer())
    # sess.run(tf.global_variables_initializer())
    # coord = tf.train.Coordinator()
    # threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # for i in range(2):
    #     example, l = sess.run([img_batch, label_batch])
    #     print l
    # coord.request_stop()
    # coord.join(threads)


if __name__ == '__main__':
    # tf.app.run()
    # generate_test_valid_train_set()
    # train_image_names, valid_image_names, train_labels, valid_labels = train_test_split(train_valid_image_names,
    #                                                                                     train_valid_labels,
    #                                                                                     test_size=0.15,
    #                                                                                     shuffle=True)
    # create_record(train_image_names, train_labels, "train.tfrecords")
    # create_record(valid_image_names, valid_labels, "valid.tfrecords")
    # create_record(test_image_names, test_labels, "test.tfrecords")

    # train_image_num = 5094
    # valid_image_num = 900
    # test_image_num = 5794
    # train_batch_size = 64
    # valid_batch_size = 64
    # test_batch_size = 64
    # train_epoches = 20

    images = tf.placeholder(tf.float32, [None, 224, 224, 3])
    labels = tf.placeholder(tf.int64, [None])
    train_mode = tf.placeholder(tf.bool)

    vgg = vgg19.Vgg19('./vgg19.npy')
    vgg.build(images, train_mode)

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=vgg.fc8))
    train = tf.train.AdagradOptimizer(0.0001).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(vgg.fc8, 1), labels)
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # num_correct_preds = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

    start = time.time()

    train_image_batch, train_label_batch = get_batch("train.tfrecords", 64)
    # valid_image_batch, valid_label_batch = get_batch("valid.tfrecords", valid_batch_size)
    # test_image_batch, test_label_batch = get_batch("test.tfrecords", test_batch_size)

    with tf.Session() as sess:
        print "Initialize Variables"
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        try:
            # train_batch = int(train_image_num / train_batch_size)
            # valid_batch = int(valid_image_num / valid_batch_size)
            # test_batch = int(test_image_num / test_batch_size)
            for i in range(5):
                if coord.should_stop():
                    break
                for j in range(30):
                    train_image, train_label = sess.run([train_image_batch, train_label_batch])
                    _, batch_loss = sess.run([train, loss],
                                             feed_dict={images: train_image, labels: train_label, train_mode: True})
                    print ("Epoch: {} step: {} loss: {} time: {} seconds".format(i, j, batch_loss, time.time() - start))
                    # print ("Training Accuracy: {}".format(
                    #     accuracy.eval(feed_dict={images: train_image, labels: train_label, train_mode: True})))

                train_loss = 0.0
                train_correct_num = 0

                # for j in range(train_batch + 1):
                #     train_image, train_label = sess.run([train_image_batch, train_label_batch])
                #     train_batch_correct_num, train_batch_loss = sess.run([num_correct_preds, loss],
                #                                                          feed_dict={images: train_image,
                #                                                                     labels: train_label,
                #                                                                     train_mode: False})
                #     train_loss += train_batch_loss
                #     train_correct_num += train_batch_correct_num
                #
                # print
                # print ("Epoch: {}".format(i))
                # print ("Train Loss: {}".format(train_loss))
                # print ("Correct_train_count: {}  Total_train_count: {}".format(train_correct_num, train_image_num))
                # print ("Train Data Accuracy: {}".format(100.0 * train_correct_num / (1.0 * train_image_num)))
                # print
                #
                # valid_loss = 0.0
                # valid_corrent_num = 0

                # for j in range(valid_batch + 1):
                #     valid_image, valid_label = sess.run([valid_image_batch, valid_label_batch])
                #     valid_batch_correct_num, valid_batch_loss = sess.run([num_correct_preds, loss],
                #                                                          feed_dict={images: valid_image,
                #                                                                     labels: valid_label,
                #                                                                     train_mode: False})
                #     valid_loss += valid_batch_loss
                #     valid_corrent_num += valid_batch_correct_num
                #
                # print
                # print ("Epoch: {}".format(i))
                # print ("Validation Loss: {}".format(valid_loss))
                # print ("Correct_val_count: {}  Total_val_count: {}".format(valid_corrent_num, valid_image_num))
                # print ("Validation Data Accuracy: {}".format(100.0 * valid_corrent_num / (1.0 * valid_image_num)))
                # print
                #
                # if (i + 1) % 20 == 0:
                #     test_loss = 0.0
                #     test_correct_num = 0
                #     for j in range(test_batch + 1):
                #         test_image, test_label = sess.run([test_image_batch, test_label_batch])
                #         test_batch_correct_num, test_batch_loss = sess.run([num_correct_preds, loss],
                #                                                            feed_dict={images: test_image,
                #                                                                       labels: test_label,
                #                                                                       train_mode: False})
                #         test_loss += test_batch_loss
                #         test_correct_num += test_batch_correct_num
                #
                #     print
                #     print ("Epoch: {}", i)
                #     print ("Test Loss: {}".format(test_loss))
                #     print ("Correct_test_count: {}  Total_test_count: {}".format(test_correct_num, test_image_num))
                #     print ("Test Data Accuracy: {}".format(100.0 * test_correct_num / (1.0 * test_image_num)))
                #     print

        except tf.errors.OutOfRangeError:
            print('Done!')
        finally:
            coord.request_stop()
        coord.join(threads)


