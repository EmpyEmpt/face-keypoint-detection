import csv
import numpy as np
import pandas as pd
import config as cfg
import cv2
from PIL import Image
import tensorflow as tf
import model as md


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
    """Split dataset (pandas df) into training and testing subsets"""
    test_indices = np.random.rand(len(dataset)) < test_ratio
    return dataset[~test_indices], dataset[test_indices]


def crop_face(image, x, y, w, h):
    image = image[y:y + h, x:x + w]
    return image


def prep_image(dp, size=cfg.CROP_SIZE, images_path=cfg.IMAGES_PATH):
    """Datapoint as list -> Image   - np array (size, size, 3))
                            Points  - tensor (136, )
    Image preparation for use in training/validation
    image as X, points as Y"""
    wh = int(dp[2])
    points = [(float(a) / wh) for a in dp[3]]

    # crop out face from image
    image = cv2.imread(images_path + dp[0])
    image = image[int(dp[1][1]):int(dp[1][1]) + wh,
                  int(dp[1][0]):int(dp[1][0]) + wh]
    # image = crop_face(image, int(dp[1][0]), int(dp[1][1]), wh, wh)

    # resize image to size * size
    image = Image.fromarray(image)
    image = image.resize((size, size))
    image = np.array(image)
    points = tf.convert_to_tensor(points, dtype=tf.float32)

    return image, points


def train_model(labels_path=cfg.LABELS_PATH, checkpoint_path=cfg.CHECKPOINT_PATH, size = cfg.CROP_SIZE):
    ds = read_whole_csv(labels_path)
    ds = pd.DataFrame(ds)
    train_ds, test_ds = split_dataset(ds)

    trainX = []
    trainY = []
    testX = []
    testY = []

    for i in range(train_ds.shape[0]):
        image, points = prep_image(train_ds.iloc[i], size)
        input_image = tf.cast(image, dtype=tf.int32)
        trainX.append(input_image)
        trainY.append(points)

    for i in range(test_ds.shape[0]):
        image, points = prep_image(train_ds.iloc[i], size)
        input_image = tf.cast(image, dtype=tf.int32)
        testX.append(input_image)
        testY.append(points)

    trainX = tf.convert_to_tensor(trainX, dtype=tf.float32) / 255
    trainY = tf.convert_to_tensor(trainY, dtype=tf.float32)
    testX = tf.convert_to_tensor(testX, dtype=tf.float32) / 255
    testY = tf.convert_to_tensor(testY, dtype=tf.float32)

    model = md.compile_model()

    batch_size = 78
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True,
        save_freq=5*batch_size)

    model.fit(trainX,
              trainY,
              epochs=1,
              validation_data=(testX, testY),
              shuffle=True,
              callbacks=[cp_callback])
    return model
