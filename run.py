import tensorflow as tf
import numpy as np
import config as cfg
import csv
import pandas as pd
from PIL import Image
import os
import cv2

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K


class image:
    filename: str
    width_height: int
    TL: list
    BR: list
    points: list

    def __init__(self, fn, tlx, tly, brx, bry, points):
        # print(f'fucking {fn}')
        self.filename = fn
        self.TL = [tlx, tly]
        self.BR = [brx, bry]
        self.points = points
        self.width_height = self.BR[0] - self.TL[0]

    def to_list(self):
        return [self.filename, self.TL, self.width_height, self.points]

    def read_all(path):
        file = open(path)
        csvreader = csv.reader(file)
        all = []
        i = 0
        for row in csvreader:
            if i == 0:
                i += 1
                continue
            ye = True
            for value in row:
                if value.startswith('-'):
                    ye = False
                    break
            if ye:
                all.append(image(row[0], int(row[1]), int(
                    row[2]), int(row[3]), int(row[5]), row[5:]).to_list())
        file.close()
        return all


# def normalize(input_image):
#     # TODO: input datapoint (csv)
#     input_image = tf.cast(input_image, tf.float32) / 255.0
#     return input_image


# def load_image(datapoint):
#     # TODO: input datapoint (csv)
#     input_image = tf.image.resize(datapoint['image'], (128, 128))
#     input_image = normalize(input_image)
#     return input_image


def split_dataset(dataset, test_ratio=0.20):
    """Splits a panda dataframe in two."""
    test_indices = np.random.rand(len(dataset)) < test_ratio
    return dataset[~test_indices], dataset[test_indices]


def crop_face(image, image_path, x, y, w, h):
    roi_color = image[y:y + h, x:x + w]
    return roi_color


def prep_image(dp):
    image_path = dp[0]
    bb = dp[1]
    wh = dp[2]
    points = dp[3]
    # crop out face from image
    image = cv2.imread(cfg.IMAGES_PATH + image_path)
    image = crop_face(image, image_path, int(
        bb[0]), int(bb[1]), int(wh), int(wh))
    path = 'data\data\\tmp\\' + image_path
    cv2.imwrite(path, image)

    # resize image to xxx * xxx
    image = Image.open(path)
    image = image.resize((194, 194))

    path = 'data\data\\croped\\' + image_path
    image.save(path)

    # save resize ratio
    crop_r = cfg.CROP_SIZE / wh

    # resize points using resize ratio
    for point in points:
        point = float(point)
        point = point * crop_r

    # convert points to relative [0.0 -> 1]
    npo = []
    for point in points:
        point_ = float(point) / wh
        npo.append(point_)

    # (should I really convert it to tensor?)
    npo = tf.convert_to_tensor(npo, dtype=tf.float32)

    # PROFIT?
    return path, npo


def build():
    # initialize the model along with the input shape to be
    # "channels last" and the channels dimension itself
    model = Sequential()
    inputShape = (194, 194, 3)
    # chanDim = -1
    # if we are using "channels first", update the input shape
    # and channels dimension
    # if K.image_data_format() == "channels_first":
    #     inputShape = (3, 194, 194)
    #     chanDim = 1
    # CONV => RELU => POOL
    model.add(Conv2D(32, (3, 3), padding="same",
                     input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))
    # (CONV => RELU) * 2 => POOL
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    # (CONV => RELU) * 2 => POOL
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(Dropout(0.25))
    # first (and only) set of FC => RELU layers
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    # softmax classifier
    model.add(Dense(136))
    # return the constructed network architecture
    return model


def main():
    input_shape = (194, 194, 3)
    output_shape = (136, )
    ds = image.read_all(cfg.LABELS_PATH)
    ds = pd.DataFrame(ds)
    train_ds, test_ds = split_dataset(ds)

    # image 194x194: tensor of shape (194, 194, 3) int32,
    # points normalized to [0.0 -> 1.0]: tensor of shape (68, 2) float32
    trainX = []
    trainY = []
    testX = []
    testY = []

    # for i in range(train_ds.shape[0]):
    for i in range(train_ds.shape[0]):
        path, points = prep_image(train_ds.iloc[i])
        input_image = tf.io.read_file(path)
        input_image = tf.image.decode_image(input_image)
        input_image = tf.cast(input_image, dtype=tf.int32)
        trainX.append(input_image)
        trainY.append(points)

        # VERY TEMPOPARY BREAK
        # break

    for i in range(test_ds.shape[0]):
        path, points = prep_image(train_ds.iloc[i])
        input_image = tf.io.read_file(path)
        input_image = tf.image.decode_image(input_image)
        input_image = tf.cast(input_image, dtype=tf.int32)
        testX.append(input_image)
        testY.append(points)

        # VERY TEMPOPARY BREAK
        # break
    trainX_ = tf.convert_to_tensor(trainX, dtype=tf.int32)
    trainY_ = tf.convert_to_tensor(trainY, dtype=tf.float32)

    testX_ = tf.convert_to_tensor(testX, dtype=tf.int32)
    testY_ = tf.convert_to_tensor(testY, dtype=tf.float32)
    checkpoint_path = "training_1/cp-{epoch:04d}.ckpt"

    checkpoint_dir = os.path.dirname(checkpoint_path)
    batch_size = 32

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True,
        save_freq=5*batch_size)

    # model = SmallerVGGNet.build()
    model = build()

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.MeanAbsoluteError(),
                  metrics=['accuracy'])
    model.fit(trainX_,
              trainY_,
              epochs=50,
              batch_size=batch_size,
              validation_data=(testX_, testY_),
              callbacks=[cp_callback])
    model.save('saved_model/model_1')


if __name__ == '__main__':
    main()
