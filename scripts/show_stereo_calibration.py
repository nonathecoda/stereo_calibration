from icecream import ic
import numpy as np
import cv2
from matplotlib import pyplot as plt


# Camera parameters to undistort and rectify images
cv_file = cv2.FileStorage()
cv_file.open('/Users/antonia/dev/masterthesis/stereo_calibration/calibration_data/stereoMap1.xml', cv2.FileStorage_READ)

stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()

frame_right = cv2.imread("images/charuco_5x5_art_people/right/2021-11-15_22-51-57.jpg", 0)
frame_left = cv2.imread("images/charuco_5x5_art_people/left/2021-11-15_22-51-57.jpg", 0)

ic(frame_right.shape)
ic(frame_left.shape)


# Undistort and rectify images
frame_right = cv2.remap(frame_right, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
frame_left = cv2.remap(frame_left, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

plt.imshow(frame_right,'gray')
plt.imshow(frame_left,'gray', alpha=0.5)
plt.show()
exit()
stereo = cv2.StereoBM.create(numDisparities=96, blockSize=5)
stereo.setTextureThreshold(10)
disparity = stereo.compute(frame_left,frame_right)

disparity[disparity == 0] = 0.1  # or a small positive value close to zero

# Calculate the depth map
focal_length = 0.3528
baseline = 171.60618066
depth_map = (focal_length * baseline) / disparity


plt.imshow(depth_map)
plt.show()