from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import random
from random import randint
from PIL import Image, ImageEnhance


def gray2rgb(img):
    if len(img.shape) < 3:
        img = np.stack((img,) * 3, axis=2)
    return img


def random_rotation(image, mode=Image.BICUBIC):
    random_angle = np.random.randint(1, 360)
    return image.rotate(random_angle, mode)


def add_random_noise(img):
    return img + np.random.normal(0, 20.0, (img.shape))


def rgb_mean_normalize(img):
    img[:, :, 0] -= 123.68
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 103.94
    return img


def random_crop(img, width, height):
    width1 = randint(0, img.size[0] - width)
    height1 = randint(0, img.size[1] - height)
    width2 = width1 + width
    height2 = height1 + height
    img = img.crop((width1, height1, width2, height2))
    return img


def random_flip_left_right(img):
    prob = randint(0, 1)
    if prob == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def random_contrast(img, lower=0.2, upper=1.8):
    factor = random.uniform(lower, upper)
    img = ImageEnhance.Sharpness(img)
    img = img.enhance(factor)
    return img


def random_brightness(img, lower=0.6, upper=1.4):
    factor = random.uniform(lower, upper)
    img = ImageEnhance.Brightness(img)
    img = img.enhance(factor)
    return img


def random_color(img, lower=0.6, upper=1.5):
    factor = random.uniform(lower, upper)
    img = ImageEnhance.Color(img)
    img = img.enhance(factor)
    return img


def per_image_standardization(img):
    if img.mode == 'RGB':
        channel = 3
    num_compare = img.size[0] * img.size[1] * channel
    img_arr = np.array(img)
    img_t = (img_arr - np.mean(img_arr)) / max(np.std(img_arr), 1 / num_compare)
    return img_t


# def preprocess(image):
#     rgb_means = [123.68, 116.78, 103.94]
#     image = tf.image.decode_jpeg(image, channels=3)
#     width, height = 1024, 768
#     image = tf.expand_dims(image, 0)
#     image = tf.image.resize_bilinear(image, [height / 2, width / 2], align_corners=False)
#     image = tf.squeeze(image)
#     image = tf.to_float(image)
#     image = tf.image.random_flip_left_right(image)
#     channels = tf.split(axis=2, num_or_size_splits=3, value=image)
#     for i in range(3):
#         channels[i] -= rgb_means[i]
#     return tf.concat(axis=2, values=channels)