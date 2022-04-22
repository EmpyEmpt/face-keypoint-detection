import csv
import numpy as np
import pandas as pd
import config as cfg
import cv2
from PIL import Image
import tensorflow as tf

# CODE'S BELOW THE BIG DIRTY EXPLANATION
"""Explaining what this file is about:
1. We somehow need to get the landmarks themself, so the net would know WHAT to search for
    And obviously we don't want to mark A LOT (15k in this dataset) of images ourself
2. We need to store it somehow

And thus this was created

It stands on a pretty simple (but kind of stupid tbh) logic:

If we need to create a new dataset - we can use some existing model as a reference to 
recreate its results (but with our model!)
For this we use create_splits: 

(this should be done in advance, it takes a lot of time and scales linearly with the amount of photos)
0. Use images_to_csv to run predictions of REFERENCE and store them in .csv 
    Which'll have strings like 'filename - boundbox coordinates (2 x,y pairs) - (x,y) pairs x68 (as much as reference has)'
    Since we save all this info in text - we can reuse it! And/or add more easily if we wish

1.  Whenever we need to create the create a new (version of) dataset - we simply use this .csv and our images
    read_whole_csv does just this - 'unzips' .csv to get all info on perline basis

2. We need to split the data - validation of the model should't be done on the data it has seen
    split_dataset does it
    As the name says - it splits our dataset into two

3. We need to crop out the faces and adjust the points coordinates
    Model DOES NOT DETECT nor CROP the image, we have to do it in advance
    prep_image plays here:
        It takes the row gotten after step 1, 
        crops the face from it,
        resizes it to the needed size (since model always expects input in the same shape), say 128x128x3
        recalculates positions of landmarks to be relative to cropped image (face)
        and packages BOTH image and points into np.array and tf.tensor
    So now we should have two pairs of x y splitted datasets
    X is what we feed the model
    Y is what we expect to recieve

    train* used for training
    test* used for validation

5. Since we dont want to redo everything everytime we come back - we have to store those 'crops'
    compress_splits is here!:
    Simply give you four-peace-dataset and it will save it all in .npz format

And whenever we need to retrieve the dataset:
    uncompress_splits just unzipz those .npz files to get back that four-peace-dataset

fetch_splits is basically a big simplification of everything and does every step itself
"""


def read_whole_csv(path):
    """Reading csv for further use after preposeccing image
    with 'image to csv' 
    Implementation ignores images which have negative values
    """
    # (as for I'm stupid it is so)
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
    Either uncompresses from .npz files or creates new ones"""

    if create_new:
        trainx, trainy, testx, testy = create_splits(
            labels, size=size, dataset_path=dataset_path)
    else:
        trainx, trainy, testx, testy = uncompress_splits()
    return trainx, trainy, testx, testy
