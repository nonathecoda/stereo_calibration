from icecream import ic
import numpy as np
import cv2


# Camera parameters to undistort and rectify images
cv_file = cv2.FileStorage()
cv_file.open('/Users/antonia/dev/masterthesis/stereo_calibration/calibration_data/stereoMap.xml', cv2.FileStorage_READ)

stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()

frame_right = cv2.imread("images/test/right/test.jpg")
frame_left = cv2.imread("images/test/left/test.jpg")

ic(frame_right.shape)
ic(frame_left.shape)


# Undistort and rectify images
frame_right = cv2.remap(frame_right, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
frame_left = cv2.remap(frame_left, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
                    
# Show the frames
cv2.imshow("frame right", frame_right) 
cv2.imshow("frame left", frame_left)
cv2.waitKey(0)
cv2.destroyAllWindows()