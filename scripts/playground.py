from icecream import ic
import cv2
import numpy as np
import cv2, PIL, os
from cv2 import aruco
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
import glob

ARUCO_DICT = cv2.aruco.DICT_4X4_250
SQUARES_VERTICALLY = 26
SQUARES_HORIZONTALLY = 12
SQUARE_LENGTH = 0.1
MARKER_LENGTH = 0.075
LENGTH_PX = int(SQUARE_LENGTH*SQUARES_VERTICALLY*1000)  # total length of the page in pixels
MARGIN_PX = 0    # size of the margin in pixels
SAVE_NAME = '/Users/antonia/dev/masterthesis/stereo_calibration/ChArUco_Marker_large.png'
# ------------------------------

aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
board = cv2.aruco.CharucoBoard((SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), SQUARE_LENGTH, MARKER_LENGTH, aruco_dict)
board.setLegacyPattern(True)
size_ratio = SQUARES_HORIZONTALLY / SQUARES_VERTICALLY
img = cv2.aruco.CharucoBoard.generateImage(board, (LENGTH_PX, int(LENGTH_PX*size_ratio)), marginSize=MARGIN_PX)

cv2.imshow("test", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite(SAVE_NAME, img)