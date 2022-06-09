import tensorflow as tf
import cv2
import dlib


def predict_image(model, image_path: str, save_path: str):
    size = model.input_shape[1]
    # load
    image = cv2.imread(image_path)

    # might want to localize face in here
    # whould also help find multiple faces
    # ...

    # resize
    res_factor_x = size / image.shape[0]
    res_factor_y = size / image.shape[1]

    input_image = cv2.resize(image, (size, size),
                             interpolation=cv2.INTER_AREA) / 255
    input_image = tf.convert_to_tensor(input_image, dtype=tf.float32)
    input_image = tf.expand_dims(input_image, axis=0)

    # predict
    res = model(input_image)[0].numpy() * 192

    # visualize
    for (x, y) in res:
        image = cv2.circle(image, (int(x/res_factor_x),
                           int(y/res_factor_y)), 1, (0, 255, 0), 5)

    # save
    if save_path is not None:
        cv2.imwrite(save_path, image)
        return
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# might not work now
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
