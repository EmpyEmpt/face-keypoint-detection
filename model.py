import tensorflow as tf
import numpy as np
import pandas as pd
import config as cfg
import csv


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
    cropw: int
    croph: int
    TL: list
    BR: list
    points: list

    def __init__(self, fn, tlbr, points):
        # print(f'fucking {fn}')
        self.filename = fn
        self.TL = tlbr[0]
        self.BR = tlbr[1]
        self.points = points
        # self.cropw = self.TL[0] - self.BR[0]
        # self.croph = self.TL[1] - self.BR[1]
        print(type(tlbr[0]), type(tlbr), type(
            self.TL), type(points), type(points[0]))

    def prep_points(self, lst):
        pass

    def readall(path):
        file = open(path)
        csvreader = csv.reader(file)
        all = []
        for row in csvreader:
            all.append(image(row[0], row[1], row[2:]))
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


def main():
    # df = pd.read_csv(cfg.LABELS_PATH)
    # train_ds_pd, test_ds_pd = split_dataset(df)
    # TRAIN_LENGTH = 1  # tmp
    # BATCH_SIZE = 64
    # BUFFER_SIZE = 1000
    # STEPS_PER_EPOCH = TRAIN_LENGTH
    # # train_images = dataset['train'].map(
    # #     load_image, num_parallel_calls=tf.data.AUTOTUNE)
    # # test_images = dataset['test'].map(
    # #     load_image, num_parallel_calls=tf.data.AUTOTUNE)
    # input_shape = (128, 128, 3)
    # output_shape = (1, 67)
    rows = image.readall(cfg.LABELS_PATH)
    print(rows[5])


if __name__ == '__main__':
    main()
