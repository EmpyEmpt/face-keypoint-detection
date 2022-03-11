from face_finder import crop_face
import tensorflow as tf
import numpy as np
import config as cfg
import csv
import pandas as pd
from PIL import Image
import cv2


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

    # crop out face from image
    imagepath = cv2.imread(cfg.IMAGES_PATH + image)
    _, path = crop_face(imagepath,
                        int(bb[0]), int(bb[1]), int(wh), int(wh))

    # resize image to xxx * xxx
    image = Image.open(path)
    image = image.resize((194, 194))
    image.save(path)

    # save resize ratio
    crop_r = cfg.CROP_SIZE / wh

    # resize points using resize ratio
    for point in points:
        point = float(point)
        point = point * crop_r

    # convert points to relative [0.0 -> 1]
    npo = []
    # for i in range(0, 135, 2):
    #     point1 = float(points[i]) / wh
    #     point2 = float(points[i+1]) / wh
    #     npo.append([point1, point2])
    for point in points:
        point_ = float(point) / wh
        npo.append(point_)

    # (should I really convert it to tensor?)
    npo = tf.convert_to_tensor(npo, dtype=tf.float32)

    # PROFIT?
    return path, npo


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
    for i in range(2):
        path, points = prep_image(train_ds.iloc[i])
        input_image = tf.io.read_file(path)
        input_image = tf.image.decode_jpeg(input_image)
        input_image = tf.cast(input_image, dtype=tf.int32)
        trainX.append(input_image)
        trainY.append(points)

        # VERY TEMPOPARY BREAK
        break

    # for i in range(test_ds.shape[0]):
    for i in range(2):
        path, points = prep_image(train_ds.iloc[i])
        input_image = tf.io.read_file(path)
        input_image = tf.image.decode_jpeg(input_image)
        input_image = tf.cast(input_image, dtype=tf.int32)
        testX.append(input_image)
        testY.append(points)

        # VERY TEMPOPARY BREAK
        break


if __name__ == '__main__':
    main()
