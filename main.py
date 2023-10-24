""" Alexandre Hachet 2023

Script that read a video from a simple/double pendulum experiment and extract all the points 

HSV detection points: 85 and 105

"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import *

def get_range(wl):
    return [(np.array([w-5, 100, 140]), np.array([w+5, 255, 255])) for w in wl]

def detect_func(thresh):
    x, y = np.where(np.all(thresh!=[240, 240, 240],axis=2))
    return np.mean(x), np.mean(y)

def detect(filename, wl, detect_f, show_masks=False):
    cap = cv2.VideoCapture(filename)
    vid_len = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    wl_range = get_range(wl)

    det = {}
    for i in range(len(wl)):
        det[i] = []
    i = 0
    
    while (1):
        _, frame = cap.read()
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        except:
            break

        results = []
        for wl_ in wl_range:
            mask = cv2.inRange(hsv, wl_[0], wl_[1])
            results.append(cv2.bitwise_and(frame, frame, mask = mask))

        if i > 0:
            threshs = []
            for j in range(len(results)):
                thresh = cv2.threshold(results[j], 230, 240, cv2.THRESH_BINARY)[1]
                det[j].append(detect_f(thresh))
                threshs.append(cv2.dilate(thresh, None, iterations=2))

            if show_masks:
                cv2.imshow("r1", threshs[0])
                cv2.imshow("r2", threshs[1])
                cv2.imshow("vid", frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

        i += 1
        print(f"Progress: {i} of {int(vid_len)}", end="\r")

    if show_masks: 
        cv2.destroyAllWindows()
        cap.release()

    return det


print(detect("IMG_2211.MOV", [85, 105], detect_func))
