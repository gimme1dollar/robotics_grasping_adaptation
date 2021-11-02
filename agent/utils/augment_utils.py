'''
    Code modified from https://hagler.tistory.com/189
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import random

def rotation(img, angle):
    angle = int(random.uniform(-angle, angle))
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))
    return img

def horizontal_flip(img, flag):
    if flag:
        return cv2.flip(img, 1)
    else:
        return img

def vertical_flip(img, flag):
    if flag:
        return cv2.flip(img, 0)
    else:
        return img


if __name__ == "__main__":
    images = sorted(glob.glob('example.png'))

    for fname in images:
        img = cv2.imread(fname)
        img = cv2.resize(img, dsize=(88, 88),interpolation=cv2.INTER_LINEAR)
        img = rotation(img, 180)
        #if random.uniform(0,1) > 0.5:
        #    img = vertical_flip(img, 1)
        file_name = dir + str(i) + '.png'
        #file_name = 'aug_image/' + str(i) + '.png'
        cv2.imwrite(file_name, img)
        i = i + 1
        if i > 9500:
            break
