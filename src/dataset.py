import numpy as np
import tensorflow as tf
import numpy as np
import albumentations as A
import cv2
import dlib
import os
from imutils import face_utils
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


transform = A.Compose(
    [A.Rotate(p=0.6, limit=15),
     A.HorizontalFlip(p=0.5),
     A.ImageCompression(quality_lower=20, quality_upper=70, p=1),
     A.GaussianBlur(blur_limit=(3, 13), sigma_limit=0, p=0.8),
     A.RandomBrightnessContrast(p=0.4)
     ],
    keypoint_params=A.KeypointParams(
        format='xy', remove_invisible=False)
)


def augment(image, keypoint):
    transformed = transform(image=image, keypoints=keypoint)
    image = transformed['image']
    image = np.array(image, dtype=np.uint8)
    keypoint = transformed['keypoints']
    return image, keypoint


def reshape(image, keypoint, image_size):
    keypoint = np.array(keypoint, dtype=np.int16) / image.shape[0] * image_size
    keypoint = keypoint.astype(dtype=np.uint8)

    image = cv2.resize(image, (image_size, image_size),
                       interpolation=cv2.INTER_AREA)


def create_dataset(dataset_path: str, image_size: int, maxi=np.Infinity):
    images = np.empty([0, image_size, image_size, 3], dtype=np.uint8)
    keypoints = np.empty([0, 68, 2], dtype=np.int16)

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
            keypoint = face_utils.shape_to_np(shape)

            image, keypoint = augment(image, keypoint)
            image, keypoint = reshape(image, keypoint, image_size)

            image = np.expand_dims(image, axis=0)
            shape = np.expand_dims(shape, axis=0)

            images = np.append(images, image, axis=0)
            keypoints = np.append(keypoints, shape, axis=0)

            break
        co += 1
        if co > maxi:
            break
    return images, keypoints
