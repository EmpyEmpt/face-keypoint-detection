"""Dataset prepation: pictures to dots"""
from imutils import face_utils
import dlib
import cv2
import os
import config as cfg
import pandas as pd

d = []

p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)
directory = os.fsencode(cfg.DATASET_PATH)

print("STARTING")
for file in os.listdir(directory):
    a = {}
    filename = os.fsdecode(file)
    image = cv2.imread(cfg.DATASET_PATH + filename)

    print('WORKING ON ', filename)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        a['filename'] = filename
        a['rect'] = [[rect.tl_corner().x, rect.tl_corner().y], [
            rect.br_corner().x, rect.br_corner().y]]
        for i in range(68):
            l = list(shape[i])
            a[i] = [l[0] - a['rect'][0][0], l[1] - a['rect'][0][1]]
        d.append(a)

df = pd.DataFrame(data=d)
print('DONE')
df.to_csv('data\\data\\relativeCoords.csv', index=False)
