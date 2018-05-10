import os
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split

path = os.getcwd()

dict_train_test_split = {}
train_valid_image_names, train_valid_labels = [], []
train_image_names, train_labels = [], []
valid_image_names, valid_labels = [], []
test_image_names, test_labels = [], []


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
                test_labels.append(int(l[1])-1)
            else:
                train_valid_labels.append(int(l[1])-1)


def create_record(image_names, imgae_labels, out_name):
    writer = tf.python_io.TFRecordWriter(out_name)
    for index, name in enumerate(image_names):
        img_path = name
        img = Image.open(img_path)
        img = img.resize((224, 224))
        img_raw = img.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[imgae_labels[index]])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))
        writer.write(example.SerializeToString())
    writer.close()


def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename], num_epochs=10)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features={
        'label': tf.FixedLenFeature([], tf.int64),
        'img_raw': tf.FixedLenFeature([], tf.string),
    })
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [224, 224, 3])
    img = tf.cast(img, tf.float32) * (1. / 255.0)
    label = tf.cast(features['label'], tf.int32)
    print img, label

    return img, label


if __name__ == '__main__':
    generate_test_valid_train_set()
    train_image_names, valid_image_names, train_labels, valid_labels = train_test_split(train_valid_image_names,
                                                                                        train_valid_labels,
                                                                                        test_size=0.15,
                                                                                        shuffle=True)
    create_record(train_image_names, train_labels, "train.tfrecords")
    create_record(valid_image_names, valid_labels, "valid.tfrecords")
    create_record(test_image_names, test_labels, "test.tfrecords")

    # for j in range(3):
    #     img, label = read_and_decode("train.tfrecords")
    #     img_batch, label_batch = tf.train.shuffle_batch([img, label],
    #                                                     batch_size=3, capacity=2000,
    #                                                     min_after_dequeue=1000)
    #     init = tf.initialize_all_variables()
    #     with tf.Session() as sess:
    #         sess.run(init)
    #         coord = tf.train.Coordinator()
    #         threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    #
    #         for i in range(2):
    #             example, l = sess.run([img_batch, label_batch])
    #         coord.request_stop()
    #         coord.join(threads)
