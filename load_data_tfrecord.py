import tensorflow as tf
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def write_data(file_name, data_path, label_path):
    writer = tf.python_io.TFRecordWriter(file_name + ".tfrecords")
    for i in range(len(data_path)):
        img_path = "jpg/" + data_path[i]
        img0 = Image.open(img_path)
        img = img0.resize((50, 50))
        img_raw = img.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label_path[i]])),
            "img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))
        writer.write(example.SerializeToString())
        if file_name == "train":
            random_angle = np.random.randint(1, 360)
            img0.rotate(random_angle)
            img = img0.resize((50, 50))
            img_raw = img.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label_path[i]])),
                "img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
            writer.write(example.SerializeToString())
            img = img0.transpose(Image.FLIP_LEFT_RIGHT)
            img = img.resize((50, 50))
            img_raw = img.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label_path[i]])),
                "img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
            writer.write(example.SerializeToString())
            img = img0.transpose(Image.FLIP_TOP_BOTTOM)
            img = img.resize((50, 50))
            img_raw = img.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label_path[i]])),
                "img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
            writer.write(example.SerializeToString())
    writer.close()


def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features={
        'label': tf.FixedLenFeature([], tf.int64), 'img_raw': tf.FixedLenFeature([], tf.string)
    })
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [50, 50, 3])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int64)
    return img, label


def get_batch(data, batch_size):
    img, label = read_and_decode(data)
    img_batch, label_batch = tf.train.shuffle_batch(
        [img, label], batch_size=batch_size, capacity=5000, min_after_dequeue=1000)
    label_batch = tf.one_hot(label_batch, 17, 1, 0)
    return img_batch, label_batch


def get_data():
    mat_path = "datasplits.mat"
    data_split = sio.loadmat(mat_path)
    file_path = "jpg/files.txt"
    images_path = []
    for line in open(file_path):
        line = line.rstrip('\n')
        images_path.append(line)
    train_img = []
    train_labels = []
    for i in data_split['trn1'][0]:
        train_img.append(images_path[i - 1])
        train_labels.append(int((i - 1) / 80))
    val_img = []
    val_labels = []
    for i in data_split['val1'][0]:
        val_img.append(images_path[i - 1])
        val_labels.append(int((i - 1) / 80))
    test_img = []
    test_labels = []
    for i in data_split['tst1'][0]:
        test_img.append(images_path[i - 1])
        test_labels.append(int((i - 1) / 80))
    return train_img, train_labels, val_img, val_labels, test_img, test_labels


train_img, train_labels, val_img, val_labels, test_img, test_labels = get_data()
TFRecord_list = ['train', 'val', 'test']
img_labels_list = [[train_img, train_labels], [val_img, val_labels], [test_img, test_labels]]
for index, TFRecord_name in enumerate(TFRecord_list):
    write_data(TFRecord_name, img_labels_list[index][0], img_labels_list[index][1])


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape,stddev=0.1))


w = init_weights([3, 3, 3, 4])
w2 = init_weights([3, 3, 4, 8])
w5 = init_weights([4 * 4 * 32, 256])
w_o = init_weights([256, 17])
b1 = init_weights([4])
b2 = init_weights([8])
b3 = init_weights([16])
b4 = init_weights([32])
b5 = init_weights([256])
b6 = init_weights([17])

def model(X, w, w2, w3, w4, w5, w_o, p_keep_conv):
    l1a = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(X, w, strides=[1, 1, 1, 1], padding='SAME'),b1))
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    l1 = tf.nn.dropout(l1, p_keep_conv)
    l2a = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l1, w2, strides=[1, 1, 1, 1], padding='SAME'),b2))
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    l2 = tf.nn.dropout(l2, p_keep_conv)

    l3a = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l2, w3, strides=[1, 1, 1, 1], padding='SAME'),b3))
    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    l4a = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l3, w4, strides=[1, 1, 1, 1], padding="SAME"),b4))
    l4 = tf.nn.max_pool(l4a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    print(l4.shape)
    l4 = tf.reshape(l4, [-1, w5.get_shape().as_list()[0]])
    l4 = tf.nn.dropout(l4, p_keep_conv)
    print("shape:", l4.shape)
    l5 = tf.nn.relu(tf.nn.bias_add(tf.matmul(l4, w5),b5))
    l5 = tf.nn.dropout(l5, p_keep_conv)
    pyx = tf.nn.bias_add(tf.matmul(l5, w_o),b6)
    print(pyx.shape)
    return pyx


BATCH_SIZE = 256
val_size = 340
TFRecord_file_list = ['train.tfrecords','val.tfrecords', 'test.tfrecords']

val_img, val_labels = get_batch(TFRecord_file_list[1], val_size)
image_batch, label_batch = get_batch(TFRecord_file_list[0], BATCH_SIZE)
tst_img, tst_labels = get_batch(TFRecord_file_list[2], val_size)
print("aaa:", image_batch.shape, label_batch.shape)

X = tf.placeholder("float", [None, 50, 50, 3])
Y = tf.placeholder("float", [None, 17])
p_keep_conv = tf.placeholder("float")
py_x = model(X, w, w2, w3, w4, w5, w_o, p_keep_conv)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
train_op = tf.train.AdamOptimizer(0.01).minimize(cost)

correct_pred = tf.equal(tf.argmax(py_x, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

tf.summary.scalar("loss", cost)
tf.summary.scalar("acc", accuracy)
merged_summary_op = tf.summary.merge_all()

with tf.Session() as sess:
    train_writer = tf.summary.FileWriter('log/train', sess.graph)
    valid_writer = tf.summary.FileWriter('log/valid')
    test_writer = tf.summary.FileWriter('log/test')
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    sess.run(tf.global_variables_initializer())
    try:
        for i in range(500):
            if coord.should_stop():
                break
            for j in range(10):
                img, label = sess.run([image_batch, label_batch])
                sess.run(train_op, feed_dict={X: img, Y: label, p_keep_conv: 0.5})
            val_i, vai_l = sess.run([val_img, val_labels])
            sess.run(train_op, feed_dict={X: val_i, Y: vai_l, p_keep_conv: 0.5})

            loss, acc, train_summary = sess.run([cost, accuracy, merged_summary_op],
                                                feed_dict={X: img, Y: label, p_keep_conv: 1})
            train_writer.add_summary(train_summary, i)
            print("train:", i, loss, acc)
            val_i, vai_l = sess.run([val_img, val_labels])
            val_loss, val_acc, valid_summary = sess.run([cost, accuracy, merged_summary_op],
                                                        feed_dict={X: val_i, Y: vai_l, p_keep_conv: 1})
            valid_writer.add_summary(valid_summary, i)
            print("val:", val_loss, val_acc)
        tst_i, tst_l = sess.run([tst_img, tst_labels])
        tst_loss, tst_acc, test_summary = sess.run([cost, accuracy, merged_summary_op],
                                                   feed_dict={X: tst_i, Y: tst_l, p_keep_conv: 1})
        print("test:", tst_loss, tst_acc)
        test_writer.add_summary(test_summary,10)
    except tf.errors.OutOfRangeError:
        print('Done!')
    finally:
        coord.request_stop()
    coord.join(threads)
