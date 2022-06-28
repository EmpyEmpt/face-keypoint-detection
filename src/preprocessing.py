def normalize_images(images):
    images /= 255
    return images


def normalize_keypoints(keypoints, image_size):
    keypoints /= image_size
    return keypoints


def preprocess(images, keypoints, image_size):
    images = normalize_images(images)
    keypoints = normalize_keypoints(keypoints, image_size)
    return images, keypoints
