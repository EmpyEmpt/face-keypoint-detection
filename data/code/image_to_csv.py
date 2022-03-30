from imutils import face_utils
import dlib
import cv2
import os
import pandas as pd


def images_to_csv(dataset_path: str, output_path: str, verbose=2):
    """Dataset prepation: pictures to dots"""
    d = []
    p = "shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)
    directory = os.fsencode(dataset_path)

    if verbose > 0:
        print("STARTING")
    for file in os.listdir(directory):
        a = {}
        filename = os.fsdecode(file)
        image = cv2.imread(dataset_path + filename)

        if verbose > 1:
            print('WORKING ON ', filename)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        for (i, rect) in enumerate(rects):
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            a['filename'] = filename
            a['TLx'] = rect.tl_corner().x
            a['TLy'] = rect.tl_corner().y
            a['BRx'] = rect.br_corner().x
            a['BRy'] = rect.br_corner().y
            for i in range(68):
                l = list(shape[i])
                a[f'{i}x'] = l[0] - a['TLx']
                a[f'{i}y'] = l[1] - a['TLy']
            d.append(a)
    df = pd.DataFrame(data=d)
    if verbose > 0:
        print('DONE')
    df.to_csv(output_path, index=False)
