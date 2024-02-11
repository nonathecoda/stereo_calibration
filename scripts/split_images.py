import numpy as np
import cv2 as cv
from icecream import ic
import os

img_folder = "images/test/before_split/"
output_folder ="images/test/"

for subdir, dirs, files in os.walk(img_folder):
    for file in files:
        img = cv.imread(os.path.join(subdir, file), 1)
        img_l = img[:, :int(img.shape[1]/2), :]
        img_r = img[:, int(img.shape[1]/2):, :]
        cv.imwrite(output_folder + "left/" + file, img_l)
        cv.imwrite(output_folder + "right/" + file, img_r)