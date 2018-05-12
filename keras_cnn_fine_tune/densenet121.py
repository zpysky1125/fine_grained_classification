# -*- coding: utf-8 -*-

from keras.optimizers import SGD
from keras.layers import Input, merge, ZeroPadding2D
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from custom_layers.scale_layer import Scale
import tensorflow as tf
from tfRecord import get_batch, get_shuffle_batch

train_image_num = 5094
valid_image_num = 900
test_image_num = 5794
train_batch_size = 64
valid_batch_size = 64
test_batch_size = 64
train_epoches = 20

img_rows, img_cols = 224, 224  # Resolution of inputs
channel = 3
num_classes = 200

train_losses, valid_losses, test_losses = [], [], []
train_accuracys, valid_accuracys, test_accuracys = [], [], []


def densenet121_model(img_rows, img_cols, color_type=1, nb_dense_block=4, growth_rate=32, nb_filter=64, reduction=0.5,
                      dropout_rate=0.0, weight_decay=1e-4, num_classes=None):
    eps = 1.1e-5

    # compute compression factor
    compression = 1.0 - reduction

    # Handle Dimension Ordering for different backends
    global concat_axis
    concat_axis = 3
    img_input = Input(shape=(img_rows, img_cols, color_type), name='data')

    # From architecture for ImageNet (Table 1 in the paper)
    nb_filter = 64
    nb_layers = [6, 12, 24, 16]  # For DenseNet-121

    # Initial convolution
    x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
    x = Convolution2D(nb_filter, 7, 7, subsample=(2, 2), name='conv1', bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv1_bn')(x)
    x = Scale(axis=concat_axis, name='conv1_scale')(x)
    x = Activation('relu', name='relu1')(x)
    x = ZeroPadding2D((1, 1), name='pool1_zeropadding')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        stage = block_idx + 2
        x, nb_filter = dense_block(x, stage, nb_layers[block_idx], nb_filter, growth_rate, dropout_rate=dropout_rate,
                                   weight_decay=weight_decay)

        # Add transition_block
        x = transition_block(x, stage, nb_filter, compression=compression, dropout_rate=dropout_rate,
                             weight_decay=weight_decay)
        nb_filter = int(nb_filter * compression)

    final_stage = stage + 1
    x, nb_filter = dense_block(x, final_stage, nb_layers[-1], nb_filter, growth_rate, dropout_rate=dropout_rate,
                               weight_decay=weight_decay)

    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv' + str(final_stage) + '_blk_bn')(x)
    x = Scale(axis=concat_axis, name='conv' + str(final_stage) + '_blk_scale')(x)
    x = Activation('relu', name='relu' + str(final_stage) + '_blk')(x)

    x_fc = GlobalAveragePooling2D(name='pool' + str(final_stage))(x)
    x_fc = Dense(1000, name='fc6')(x_fc)
    x_fc = Activation('softmax', name='prob')(x_fc)

    model = Model(img_input, x_fc, name='densenet')

    weights_path = 'imagenet_models/densenet121_weights_tf.h5'

    model.load_weights(weights_path, by_name=True)

    # Truncate and replace softmax layer for transfer learning
    # Cannot use model.layers.pop() since model is not of Sequential() type
    # The method below works since pre-trained weights are stored in layers but not in the model
    x_newfc = GlobalAveragePooling2D(name='pool' + str(final_stage))(x)
    x_newfc = Dense(num_classes, name='fc6')(x_newfc)
    x_newfc = Activation('softmax', name='prob')(x_newfc)

    model = Model(img_input, x_newfc)

    # Learning rate is changed to 0.001
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def conv_block(x, stage, branch, nb_filter, dropout_rate=None, weight_decay=1e-4):
    '''Apply BatchNorm, Relu, bottleneck 1x1 Conv2D, 3x3 Conv2D, and option dropout
        # Arguments
            x: input tensor
            stage: index for dense block
            branch: layer index within each dense block
            nb_filter: number of filters
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''
    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_' + str(branch)
    relu_name_base = 'relu' + str(stage) + '_' + str(branch)

    # 1x1 Convolution (Bottleneck layer)
    inter_channel = nb_filter * 4
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base + '_x1_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base + '_x1_scale')(x)
    x = Activation('relu', name=relu_name_base + '_x1')(x)
    x = Convolution2D(inter_channel, 1, 1, name=conv_name_base + '_x1', bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    # 3x3 Convolution
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base + '_x2_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base + '_x2_scale')(x)
    x = Activation('relu', name=relu_name_base + '_x2')(x)
    x = ZeroPadding2D((1, 1), name=conv_name_base + '_x2_zeropadding')(x)
    x = Convolution2D(nb_filter, 3, 3, name=conv_name_base + '_x2', bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def transition_block(x, stage, nb_filter, compression=1.0, dropout_rate=None, weight_decay=1E-4):
    ''' Apply BatchNorm, 1x1 Convolution, averagePooling, optional compression, dropout
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_filter: number of filters
            compression: calculated as 1 - reduction. Reduces the number of feature maps in the transition block.
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''

    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_blk'
    relu_name_base = 'relu' + str(stage) + '_blk'
    pool_name_base = 'pool' + str(stage)

    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base + '_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base + '_scale')(x)
    x = Activation('relu', name=relu_name_base)(x)
    x = Convolution2D(int(nb_filter * compression), 1, 1, name=conv_name_base, bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    x = AveragePooling2D((2, 2), strides=(2, 2), name=pool_name_base)(x)

    return x


def dense_block(x, stage, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1e-4,
                grow_nb_filters=True):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_layers: the number of layers of conv_block to append to the model.
            nb_filter: number of filters
            growth_rate: growth rate
            dropout_rate: dropout rate
            weight_decay: weight decay factor
            grow_nb_filters: flag to decide to allow number of filters to grow
    '''

    eps = 1.1e-5
    concat_feat = x

    for i in range(nb_layers):
        branch = i + 1
        x = conv_block(concat_feat, stage, branch, growth_rate, dropout_rate, weight_decay)
        concat_feat = merge([concat_feat, x], mode='concat', concat_axis=concat_axis,
                            name='concat_' + str(stage) + '_' + str(branch))

        if grow_nb_filters:
            nb_filter += growth_rate

    return concat_feat, nb_filter


if __name__ == '__main__':

    saver = tf.train.Saver()

    train_image_batch, train_label_batch = get_shuffle_batch("train.tfrecords", train_batch_size)
    test_train_image_batch, test_train_label_batch = get_batch("train.tfrecords", train_batch_size)
    valid_image_batch, valid_label_batch = get_batch("valid.tfrecords", valid_batch_size)
    test_image_batch, test_label_batch = get_batch("test.tfrecords", test_batch_size)

    model = densenet121_model(img_rows=img_rows, img_cols=img_cols, color_type=channel, num_classes=num_classes)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        try:
            train_batch = int(train_image_num / train_batch_size)
            valid_batch = int(valid_image_num / valid_batch_size)
            test_batch = int(test_image_num / test_batch_size)
            for i in range(100):

                if coord.should_stop():
                    break
                for j in range(train_batch + 1):
                    train_image, train_label = sess.run([train_image_batch, train_label_batch])
                    model.train_on_batch(train_image, train_label)

                train_loss = 0.0
                train_correct_num = 0

                for j in range(train_batch + 1):
                    test_train_image, test_train_label = sess.run([test_train_image_batch, test_train_label_batch])
                    test_train_test = model.test_on_batch(test_train_image, test_train_label)
                    train_loss += test_train_test[0]
                    train_correct_num += train_batch_size * test_train_test[1]

                print
                print ("Epoch: {}".format(i))
                print ("Train Loss: {}".format(train_loss))
                print ("Correct_train_count: {}  Total_train_count: {}".format(train_correct_num, train_batch_size * (train_batch + 1)))
                print ("Train Data Accuracy: {}".format(100.0 * train_correct_num / (1.0 * train_batch_size * (train_batch + 1))))
                print

                train_losses.append(train_loss)
                train_accuracys.append(100.0 * train_correct_num / (1.0 * train_batch_size * (train_batch + 1)))

                valid_loss = 0.0
                valid_correct_num = 0

                for j in range(valid_batch + 1):
                    valid_image, valid_label = sess.run([valid_image_batch, valid_label_batch])
                    valid_test = model.test_on_batch(valid_image, valid_label)
                    valid_loss += valid_test[0]
                    valid_correct_num += valid_batch_size * valid_test[1]

                print
                print ("Epoch: {}".format(i))
                print ("Validation Loss: {}".format(valid_loss))
                print ("Correct_val_count: {}  Total_val_count: {}".format(valid_correct_num, valid_batch_size * (valid_batch + 1)))
                print ("Validation Data Accuracy: {}".format(100.0 * valid_correct_num / (1.0 * valid_batch_size * (valid_batch + 1))))
                print

                valid_losses.append(valid_loss)
                valid_accuracys.append(100.0 * valid_correct_num / (1.0 * valid_batch_size * (valid_batch + 1)))

                if (i + 1) % 10 == 0:

                    test_loss = 0.0
                    test_correct_num = 0

                    for j in range(test_batch + 1):
                        test_image, test_label = sess.run([test_image_batch, test_label_batch])
                        test_test = model.test_on_batch(test_image, test_label)
                        test_loss += test_test[0]
                        test_correct_num += test_batch_size * test_test[1]


                    print
                    print ("Epoch: {}", i)
                    print ("Test Loss: {}".format(test_loss))
                    print ("Correct_test_count: {}  Total_test_count: {}".format(test_correct_num, test_image_num))
                    print ("Test Data Accuracy: {}".format(100.0 * test_correct_num / (1.0 * test_batch_size * (test_batch + 1))))
                    print

                    test_losses.append(test_loss)
                    test_accuracys.append(100.0 * test_correct_num / (1.0 * test_batch_size * (test_batch + 1)))

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