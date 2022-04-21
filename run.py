import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import dlib


def predict_image(model, detector, image: str, save_path=None):
    """Runs prediction on a single image
    using dlib face detector
    Expects tf2 model and path to image"""

    image = cv2.imread(image)
    size = model.input_shape[1]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    x1 = faces[0].left()
    y1 = faces[0].top()
    x2 = faces[0].right()
    y2 = faces[0].bottom()

    input_image = image[y1:y2, x1:x2]
    dimsCrop = [[x1, y1, x2-x1, y2-y1]]

    input_image = Image.fromarray(input_image)
    input_image = input_image.resize((size, size))
    input_image = np.array(input_image)

    input_image = tf.cast(input_image, dtype=tf.float32) / 255
    input_image = tf.expand_dims(input_image, axis=0)
    res = model(input_image)

    x_ = []
    y_ = []
    for i in range(0, 136, 2):
        x_.append(
            int(res[0][i] * size * (dimsCrop[0][3] / size) + dimsCrop[0][0]))
        y_.append(int(res[0][i+1] * size *
                  (dimsCrop[0][3] / size) + dimsCrop[0][1]))

    for i in range(68):
        image = cv2.circle(image, (x_[i], y_[i]), 1,
                           (0, 0, 255), int(image.shape[1] * 0.006))

    if save_path is not None:
        cv2.imwrite(save_path, image)
        return
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def predict_stream(model):
    """Runs prediction on live webcam"""
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        frame = cv2.copyMakeBorder(frame, 50, 50, 50, 50, cv2.BORDER_CONSTANT)
        frame = predict_image(model=model, image=frame)
        cv2.imshow(winname="Face", mat=frame)
        if cv2.waitKey(delay=1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


def dlib_reference(image, detector, predictor, save_path=None):
    """Runs prediction on a single image using DLib predictor"""

    image = cv2.imread(image)

    gray = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(image=gray, box=face)
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(img=image, center=(x, y), radius=3,
                       color=(0, 255, 0), thickness=-1)

    if save_path is not None:
        cv2.imwrite(save_path, image)
        return
    return image


def run(image):
    """Runs predictions on both tf2 model and dlib"""
    model = tf.keras.models.load_model('model.h5')
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    predict_image(model, detector, image, save_path='static/mine.jpeg')
    dlib_reference(image, detector, predictor, save_path='static/dlib.jpeg')
