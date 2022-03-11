import cv2


def crop_face(image, x, y, w, h):
    roi_color = image[y:y + h, x:x + w]
    # cv2.imwrite('data\data\croped\\' + str(w) +
    #             str(h) + '_faces.jpg', roi_color)
    return roi_color, 'data\data\croped\\' + str(w) + str(h) + '_faces.jpg'


def face_finder(imagePath):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faceCascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        crop_face(image, x, y, w, h)

    return faces
