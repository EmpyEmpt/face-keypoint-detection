import numpy as np
import cv2
import tensorflow as tf
import dlib
import os
from imutils import face_utils
import albumentations as A


def create_dataset(dataset_path: str, image_size: int, maxi=np.Infinity):
    images = []
    keypoints = []
    p = "../shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)
    directory = os.fsencode(dataset_path)

    co = 0
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        image = cv2.imread(dataset_path + filename)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        for (_, rect) in enumerate(rects):
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            image = cv2.resize(image, (image_size, image_size),
                               interpolation=cv2.INTER_AREA)
            image /= 255
            shape = shape * (image_size / image.shape[0]) / image_size
            images.append(image)
            keypoints.append(shape)
            break
        co += 1
        if co > maxi:
            break
    images = tf.convert_to_tensor(images, dtype=tf.float32)
    keypoints = tf.convert_to_tensor(keypoints, dtype=tf.float32)
    return images, keypoints


def augment(images, keypoints, image_size: int):
    """This dumbass excepts on like 60%+ of images due to some cv2 error"""
    t_images = []
    t_keypoints = []

    images = (images.numpy() * 255).astype(np.int32)
    keypoints = (keypoints.numpy() * image_size).astype(np.int16)

    transform = A.Compose(
        [A.Rotate(p=0.5)],
        keypoint_params=A.KeypointParams(format='xy')
    )
    e_count = 0
    for (im, kp) in zip(images, keypoints):
        try:
            transformed = transform(image=im, keypoints=kp)
        except Exception:
            e_count += 1
            continue
        t_keypoints.append(transformed['keypoints'])
        t_images.append(transformed['image'])
    print(f'Encountered {e_count} exceptions, man Im dead')

    t_images = tf.convert_to_tensor(t_images, dtype=tf.float32) / 255
    t_keypoints = tf.convert_to_tensor(
        t_keypoints, dtype=tf.float32) / image_size

    return t_images, t_keypoints


def split_dataset(X, Y, test_ratio: float = 0.20):
    size = int(len(X) * test_ratio)
    return X[size:], X[:size], Y[size:], Y[:size]


def compress_splits(X, Y, dir: str):
    X = X.numpy()
    Y = Y.numpy()
    np.savez_compressed(dir + 'Xvalues.npz', X)
    np.savez_compressed(dir + 'Yvalues.npz', Y)


def uncompress_splits(dir: str):
    X = np.load(dir + 'Xvalues.npz')
    Y = np.load(dir + 'Yvalues.npz')

    X = X['arr_0']
    Y = Y['arr_0']

    X = tf.convert_to_tensor(X, dtype=tf.float32)
    Y = tf.convert_to_tensor(Y, dtype=tf.float32)

    return X, Y


def fetch_splits(dataset_path: str, image_size: int = 192, amount: int = np.Infinity, create_new: bool = False):
    """Returns dataset splits as tf2 tensors
    Either uncompresses from .npz files or creates new ones"""

    if create_new:
        images, keypoints = create_dataset(dataset_path, image_size, amount)
        compress_splits(images, keypoints)
    else:
        images, keypoints = uncompress_splits()

    trainx, trainy, testx, testy = split_dataset(images, keypoints)
    return trainx, trainy, testx, testy
