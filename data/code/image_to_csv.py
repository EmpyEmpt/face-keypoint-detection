"""Dataset prepation: pictures to dots"""
from imutils import face_utils
import dlib
import cv2
import os
import config as cfg
import pandas as pd

show = False
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
        a['rect'] = rect
        for i in range(68):
            a[i] = shape[i]
        d.append(a)
    if show:
        cv2.imshow("Output", gray)
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
df = pd.DataFrame(data=d)
print('DONE')
df.to_csv('data\data\DSDotted.csv', index=False)
