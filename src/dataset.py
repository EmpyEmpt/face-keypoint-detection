import numpy as np
import tensorflow as tf
from tensorflow.python.data import AUTOTUNE
from sklearn.utils import shuffle

from src.preprocessing import preprocess


def compress_splits(X, Y, dir):
    np.savez_compressed(dir + 'Xvalues.npz', X)
    np.savez_compressed(dir + 'Yvalues.npz', Y)


def uncompress_splits(dir: str):
    X = np.load(dir + 'Xvalues.npz')['arr_0']
    Y = np.load(dir + 'Yvalues.npz')['arr_0']

    return X, Y


def split_dataset(X, Y, test_ratio: float = 0.20):
    size = int(len(X) * test_ratio)
    return X[size:], X[:size], Y[size:], Y[:size]


def fetch_ds(config, op_type='train'):
    # load dataset
    images, keypoints = uncompress_splits(config['dataset']['compressed_dir'])

    # preprocess ds
    images, keypoints = preprocess(images, keypoints, config['img_shape'])

    # split ds
    images, keypoints = shuffle(images, keypoints, random_state=0)
    train_x, test_x, train_y, test_y = split_dataset(
        images, keypoints, config['dataset']['split_ratio'])

    # put into tf.ds
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_x, train_y))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y))

    # visualization
    # log_image_artifacts_to_wandb(data=train_ds, metadata=metadata)

    train_dataset = train_dataset.batch(config[op_type]['batch_size'])
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

    test_dataset = test_dataset.batch(config[op_type]['batch_size'])
    test_dataset = test_dataset.cache()
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

    return train_dataset, test_dataset
