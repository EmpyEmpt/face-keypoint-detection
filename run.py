import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import dlib


def predict_image(model, image, save_path=None):
    """Runs prediction on a single image
    Expects tf2 model and image: 
    - decoded as list
    - path as str """

    if isinstance(image, str):
        image = cv2.imread(image)
    elif not isinstance(image, list):
        print('Oi, you cant do that mate, I need em inputs bruda')
        return

    # prepare image
    size = model.input_shape[1]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faceCascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30)
    )

    dimsCrop = []
    for (x, y, w, h) in faces:
        input_image = image[y:y + h, x:x + w]
        dimsCrop.append([x, y, w, h])

    input_image = Image.fromarray(input_image)
    input_image = input_image.resize((size, size))
    input_image = np.array(input_image)

    # predict
    input_image = tf.cast(image, dtype=tf.float32) / 255
    input_image = tf.expand_dims(input_image, axis=0)
    res = model.predict(input_image)

    # put predictions on image
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
    return image


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


def dlib_reference(image, save_path=None):
    """Runs prediction on a single image 
    Uses dlib, expects image: 
    - decoded as list
    - path as str """

    if isinstance(image, str):
        image = cv2.imread(image)
    elif not isinstance(image, list):
        print('Oi, you cant do that mate, I need em inputs bruda')
        return

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
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
    predict_image(model, image=image, save_path='static/output.jpeg')
    dlib_reference(image=image, save_path='static/dlib.jpeg')
