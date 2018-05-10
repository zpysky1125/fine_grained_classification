import tensorflow as tf
import numpy as np

tfrecords_filename = 'train.tfrecords'
record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)

for string_record in record_iterator:
    # Parse the next example
    example = tf.train.Example()
    example.ParseFromString(string_record)

    # Get the features you stored (change to match your tfrecord writing code)
    img_string = int(example.features.feature['img_raw']
                                 .bytes_list
                                 .value[0])

    label = int(example.features.feature['label']
                                .int64_list
                                .value[0])

    # Convert to a numpy array (change dtype to the datatype you stored)
    img_1d = np.fromstring(img_string, dtype=np.float32)
    # Print the image shape; does it match your expectations?
    print(img_1d.shape)