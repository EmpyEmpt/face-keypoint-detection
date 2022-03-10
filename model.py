from face_finder import crop_face
import tensorflow as tf
import numpy as np
import config as cfg
import csv
import pandas as pd
from PIL import Image
import cv2


class point():
    realX: int
    realY: int
    relativeX: float
    relativeY: float

    def __init__(self, x: int, y: int):
        self.realX = x
        self.realY = y

    def __init__(self, x: float, y: float):
        self.relativeX = x
        self.relativeY = y

    def to_relative(self, w, h):
        self.relativeX = self.x / w
        self.relativeY = self.y / h

    def from_relative(self, w, h):
        self.relativeX = self.x * w
        self.relativeY = self.y * h

    def get_real(self):
        return [self.realX, self.realY]

    def get_relative(self):
        return [self.relativeX, self.relativeY]


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

    def prep_points(self, lst):
        pass

    def read_all(path):
        file = open(path)
        csvreader = csv.reader(file)
        all = []
        i = 0
        for row in csvreader:
            if i == 0:
                i += 1
                continue
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


def prep_image(dp):
    image = dp[0]
    bb = dp[1]
    wh = dp[2]
    points = dp[3]
    print('INITIAL DATA: ')
    print(image, wh, points)
    # crop out face from image
    imagepath = cv2.imread(cfg.IMAGES_PATH + image)
    _, path = crop_face(imagepath,
                        int(bb[0]), int(bb[1]), int(wh), int(wh))
    # resize image to xxx * xxx
    image = Image.open(path)
    image = image.resize((194, 194))
    image.save(path)

    # save resize ration (somehow)
    crop_r = cfg.CROP_SIZE / wh

    # resize points using resize ratio
    for point in points:
        point = float(point)
        point = point * crop_r
    # convert points to relative [0.0 -> 1]
    npo = []
    for point in points:
        point = float(point) / wh
        npo.append(point)
    print('AFTER PROCESSING: ')
    print(path, wh, npo)
    print('AND: ')
    print()
    return image, npo

    # idk, save it?


def main():
    # TRAIN_LENGTH = 1  # tmp
    # BATCH_SIZE = 64
    # BUFFER_SIZE = 1000
    # STEPS_PER_EPOCH = TRAIN_LENGTH
    # # train_images = dataset['train'].map(
    # #     load_image, num_parallel_calls=tf.data.AUTOTUNE)
    # # test_images = dataset['test'].map(
    # #     load_image, num_parallel_calls=tf.data.AUTOTUNE)
    input_shape = (128, 128, 3)
    output_shape = (1, 68*2)
    ds = image.read_all(cfg.LABELS_PATH)
    ds = pd.DataFrame(ds)
    train_ds_pd, test_ds_pd = split_dataset(ds)
    ay, npo = prep_image(train_ds_pd.iloc[1])


if __name__ == '__main__':
    main()
