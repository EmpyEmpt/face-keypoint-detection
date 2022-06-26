import numpy as np


def normalize_images(images):
    images = images.astype(np.float32) / 255
    return images


def normalize_keypoints(keypoints, image_size):
    keypoints = keypoints / image_size[0]
    return keypoints


def preprocess(images, keypoints, image_size):
    images = normalize_images(images)
    keypoints = normalize_keypoints(keypoints, image_size)
    return images, keypoints
