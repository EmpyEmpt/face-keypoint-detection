import csv
import numpy as np
import pandas as pd
import config as cfg
import cv2
from PIL import Image
import tensorflow as tf


def read_whole_csv(path):
    """Reading csv for further use after preposeccing image
    with 'image to csv' 
    Implementation ignores images which have negative values"""
    file = open(path)
    csvreader = csv.reader(file)
    all = []
    next(csvreader, None)
    for row in csvreader:
        ye = True
        for value in row:
            if value.startswith('-'):
                ye = False
                break
        if ye:
            all.append([row[0], [int(row[1]), int(row[2])],
                       int(row[3]) - int(row[1]), row[5:]])
    file.close()
    return all


def split_dataset(dataset, test_ratio=0.20):
    """Split dataset (pandas df) into two subsets"""
    test_indices = np.random.rand(len(dataset)) < test_ratio
    return dataset[~test_indices], dataset[test_indices]


def prep_image(dp, size=cfg.IMAGE_SIZE, dataset_path=cfg.DATASET_PATH):
    """Datapoint as list -> Image   - np array (size, size, 3))
                            Points  - tensor (136, )
    Image preparation for use in training/validation
    image as X, points as Y"""
    wh = int(dp[2])
    points = [(float(a) / wh) for a in dp[3]]

    # crop out face from image
    image = cv2.imread(dataset_path + dp[0])
    image = image[int(dp[1][1]):int(dp[1][1]) + wh,
                  int(dp[1][0]):int(dp[1][0]) + wh]

    # resize image to size * size
    image = Image.fromarray(image)
    image = image.resize((size, size))
    image = np.array(image)
    points = tf.convert_to_tensor(points, dtype=tf.float32)

    return image, points


def create_splits(labels=cfg.LABELS_PATH, size=cfg.IMAGE_SIZE, dataset_path=cfg.DATASET_PATH):
    """Creates new dataset splits from provided dataset"""

    ds = read_whole_csv(labels)
    ds = pd.DataFrame(ds)
    train_ds, test_ds = split_dataset(ds)

    trainx = []
    trainy = []
    testx = []
    testy = []

    for i in range(train_ds.shape[0]):
        image, points = prep_image(
            train_ds.iloc[i], dataset_path=dataset_path, size=size)
        input_image = tf.cast(image, dtype=tf.int32)
        trainx.append(input_image)
        trainy.append(points)

    for i in range(test_ds.shape[0]):
        image, points = prep_image(
            train_ds.iloc[i], dataset_path=dataset_path, size=size)
        input_image = tf.cast(image, dtype=tf.int32)
        testx.append(input_image)
        testy.append(points)

    trainx = tf.convert_to_tensor(trainx, dtype=tf.float32) / 255
    trainy = tf.convert_to_tensor(trainy, dtype=tf.float32)

    testx = tf.convert_to_tensor(testx, dtype=tf.float32) / 255
    testy = tf.convert_to_tensor(testy, dtype=tf.float32)

    return trainx, trainy, testx, testy


def compress_splits(trainx, trainy, testx, testy, dir=cfg.COMPRESSED_PATH):
    """Compresses existing dataset split as .npz"""

    trainx_ = trainx.numpy()
    trainy_ = trainy.numpy()
    testx_ = testx.numpy()
    testy_ = testy.numpy()
    np.savez_compressed(dir + 'trainx.npz', trainx_)
    np.savez_compressed(dir + 'trainy.npz', trainy_)
    np.savez_compressed(dir + 'testx.npz', testx_)
    np.savez_compressed(dir + 'testy.npz', testy_)


def uncompress_splits(dir=cfg.COMPRESSED_PATH):
    """Decompresses existing dataset split as .npz
    Returns tf2 tensors"""
    trainx = np.load(dir + 'trainx.npz')
    trainy = np.load(dir + 'trainy.npz')
    testx = np.load(dir + 'testx.npz')
    testy = np.load(dir + 'testy.npz')

    trainx = trainx['arr_0']
    trainy = trainy['arr_0']
    testx = testx['arr_0']
    testy = testy['arr_0']

    trainx = tf.convert_to_tensor(trainx, dtype=tf.float32)
    trainy = tf.convert_to_tensor(trainy, dtype=tf.float32)
    testx = tf.convert_to_tensor(testx, dtype=tf.float32)
    testy = tf.convert_to_tensor(testy, dtype=tf.float32)

    return trainx, trainy, testx, testy


def fetch_splits(labels=cfg.LABELS_PATH, dataset_path=cfg.DATASET_PATH, size=cfg.IMAGE_SIZE,  create_new=False):
    """Returns dataset splits as tf2 tensors
    Either uncompresses from .npz or creates new one"""

    if create_new:
        trainx, trainy, testx, testy = create_splits(
            labels, size=size, dataset_path=dataset_path)
    else:
        trainx, trainy, testx, testy = uncompress_splits()
    return trainx, trainy, testx, testy
