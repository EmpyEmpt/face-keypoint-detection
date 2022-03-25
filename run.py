import tensorflow as tf
import numpy as np
import config as cfg
from PIL import Image
import cv2
import train
import model


def predict_image(model, image_path,  size, image=None, save_path=None):
    if image is None:
        image = cv2.imread(image_path)
    image = cv2.copyMakeBorder(image, 50, 50, 50, 50, cv2.BORDER_CONSTANT)
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
        image = image[y:y + h, x:x + w]
        dimsCrop.append([x, y, w, h])

    image = Image.fromarray(image)
    image = image.resize((size, size))
    image = np.array(image)

    # predict
    input_image = tf.cast(image, dtype=tf.float32) / 255
    input_image = tf.expand_dims(input_image, axis=0)
    res = model.predict(input_image)
    x_ = []
    y_ = []
    for i in range(0, 136, 2):
        x_.append(
            int(res[0][i] * size * (dimsCrop[0][3] / size) + dimsCrop[0][0]))
        y_.append(int(res[0][i+1] * size *
                  (dimsCrop[0][3] / size) + dimsCrop[0][1]))

    image = cv2.imread(image_path)
    for i in range(68):
        image = cv2.circle(image, (x_[i], y_[i]), 1,
                           (0, 0, 255), int(image.shape[1] * 0.006))
    if save_path is not None:
        cv2.imwrite(save_path, image)
    return image


def predict_stream(model, size=cfg.CROP_SIZE):
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        frame = predict_image(model=model, image=frame, size=size)
        cv2.imshow(winname="Face", mat=frame)
        if cv2.waitKey(delay=1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    # model = train.train_model(size = 128)
    ml = model.load_model('models\\x128NA-89.h5')
    predict_image(ml,  'data\data\InitialDS\\0 (19).jpg',
                  size=128, save_path='pic.jpeg')


if __name__ == "__main__":
    main()
