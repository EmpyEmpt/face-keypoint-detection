import numpy as np
import dlib
import albumentations as A
import os
import cv2
from imutils import face_utils


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

def augment(image, keypoint, image_size):
    im_shape = image.shape[0]
    transformed = transform(image=image, keypoints=keypoint)
    image = transformed['image']
    image = np.array(image, dtype=np.uint8)
    keypoint = transformed['keypoints']
    keypoint = np.array(keypoint, dtype=np.int16) / \
        im_shape * image_size
    keypoint = keypoint.astype(dtype=np.uint8)
    return image, keypoint

def resize_image(image, image_size):
    image = cv2.resize(image, (image_size, image_size),
                       interpolation=cv2.INTER_AREA)
    return image

def create_dataset(config):
    dataset_path: str = config['dataset']['dataset_dir']
    image_size: int = config['img_shape'][0]
    maxi = config['amount']

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

            image, keypoint = augment(image, keypoint, image_size)

            image = resize_image(image, image_size)

            image = np.expand_dims(image, axis=0)
            images = np.append(images, image, axis=0)
            keypoint = np.expand_dims(keypoint, axis=0)
            keypoints = np.append(keypoints, keypoint, axis=0)

            break

        co += 1
        if co > maxi:
            break
    return images, keypoints
