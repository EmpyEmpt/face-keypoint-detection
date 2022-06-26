from functools import partial

import tensorflow as tf
from tensorflow.python.data import AUTOTUNE


def verify_shape(images, keypoints, img_shape, keypoint_shape):
    images = tf.ensure_shape(x=images, shape=[None, *img_shape])
    keypoints = tf.ensure_shape(x=keypoints, shape=[None, *keypoint_shape])
    return images, keypoints


def pass_tests_before_preprocessing():
    pass


def pass_tests_before_fitting(data, img_shape, keypoint_shape):
    data.map(partial(verify_shape, img_shape=img_shape,
             keypoint_shape=keypoint_shape), num_parallel_calls=AUTOTUNE)
