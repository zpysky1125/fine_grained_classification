import os

import tensorflow as tf
from sklearn.model_selection import train_test_split

path = os.getcwd()

dict_train_test_split = {}
train_valid_image_names, train_valid_labels = [], []
train_image_names, train_labels = [], []
valid_image_names, valid_labels = [], []
test_image_names, test_labels = [], []

sess = tf.Session()


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
            print l
            l = l.strip('\n').split(' ')
            if dict_train_test_split[l[0]] == 0:
                test_labels.append(int(l[1]) - 1)
            else:
                train_valid_labels.append(int(l[1]) - 1)


def create_record(image_names, image_labels, out_name):
    writer = tf.python_io.TFRecordWriter(out_name)
    for index, name in enumerate(image_names):
        print index
        img_path = name
        # img = Image.open(img_path)
        # img = img.resize((224, 224))
        # img_raw = img.tobytes()
        # example = tf.train.Example(features=tf.train.Features(feature={
        #     'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[image_labels[index]])),
        #     'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        # }))
        image_buffer, height, width = _process_image(img_path)
        example = _convert_to_example(image_buffer, image_labels[index], height, width)
        writer.write(example.SerializeToString())
    writer.close()


def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features={
        "image/encoded": tf.FixedLenFeature([], tf.string),
        "image/height": tf.FixedLenFeature([], tf.int64),
        "image/width": tf.FixedLenFeature([], tf.int64),
        "image/class/label": tf.FixedLenFeature([], tf.int64), })
    image_encoded = features["image/encoded"]
    image_raw = tf.image.decode_jpeg(image_encoded, channels=3)
    img = tf.image.resize_image_with_crop_or_pad(image_raw, 224, 224)
    label = tf.cast(features["image/class/label"], tf.int64)

    # features = tf.parse_single_example(serialized_example, features={
    #     'label': tf.FixedLenFeature([], tf.int64),
    #     'img_raw': tf.FixedLenFeature([], tf.string)
    # })
    # img = tf.decode_raw(features['img_raw'], tf.uint8)
    # img = tf.reshape(img, [224, 224, 3])
    # img = tf.cast(img, tf.float32) * (1. / 255.0) - 0.5
    # label = tf.cast(features['label'], tf.int64)
    return img, label


def get_shuffle_batch(data, batch_size):
    img, label = read_and_decode(data)
    img_batch, label_batch = tf.train.shuffle_batch(
        [img, label], batch_size=batch_size, capacity=6000, min_after_dequeue=1000, allow_smaller_final_batch=True)
    return img_batch, label_batch


def get_batch(data, batch_size):
    img, label = read_and_decode(data)
    img_batch, label_batch = tf.train.batch(
        [img, label], batch_size=batch_size, capacity=6000, allow_smaller_final_batch=True)
    return img_batch, label_batch


def main(unused_argv):
    generate_test_valid_train_set()
    train_image_names, valid_image_names, train_labels, valid_labels = train_test_split(train_valid_image_names,
                                                                                        train_valid_labels,
                                                                                        test_size=0.15,
                                                                                        shuffle=True)
    # create_record(train_image_names, train_labels, "train.tfrecords")
    # create_record(valid_image_names, valid_labels, "valid.tfrecords")
    # create_record(test_image_names, test_labels, "test.tfrecords")

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
    tf.app.run()
