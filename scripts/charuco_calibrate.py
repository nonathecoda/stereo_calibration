from icecream import ic
import cv2
import numpy as np
import cv2, PIL, os
from cv2 import aruco
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
import glob
# ------------------------------
# ENTER YOUR PARAMETERS HERE:
ARUCO_DICT = cv2.aruco.DICT_4X4_100
SQUARES_VERTICALLY = 12
SQUARES_HORIZONTALLY = 8
SQUARE_LENGTH = 0.675
MARKER_LENGTH = 0.5
LENGTH_PX = int(SQUARE_LENGTH*SQUARES_VERTICALLY*100)  # total length of the page in pixels
MARGIN_PX = 0    # size of the margin in pixels
SAVE_NAME = '/Users/antonia/dev/masterthesis/stereo_calibration/ChArUco_Marker.png'
# ------------------------------

aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
board = cv2.aruco.CharucoBoard((SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), SQUARE_LENGTH, MARKER_LENGTH, aruco_dict)
board.setLegacyPattern(True)
size_ratio = SQUARES_HORIZONTALLY / SQUARES_VERTICALLY
img = cv2.aruco.CharucoBoard.generateImage(board, (LENGTH_PX, int(LENGTH_PX*size_ratio)), marginSize=MARGIN_PX)
#cv2.imshow("img", img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#cv2.imwrite(SAVE_NAME, img)


print("POSE ESTIMATION STARTS:")
allCornersL = []
allIdsL = []
allCornersR = []
allIdsR = []
decimatorL = 0
decimatorR=0

frameSize = (3280,2464)

# SUB PIXEL CORNER DETECTION CRITERION
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

imagesLeft = sorted(glob.glob('images/charuco/left/*.jpg'))
imagesRight = sorted(glob.glob('images/charuco/right/*.jpg'))

for imgLeft, imgRight in zip(imagesLeft, imagesRight):
    print(imgLeft, imgRight)
    imgL = cv2.imread(imgLeft)
    imgR = cv2.imread(imgRight)
   
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    frameSize = grayL.shape
    
    params = cv2.aruco.DetectorParameters()
    cornersL, idsL, rejectedImgPointsL = cv2.aruco.detectMarkers(grayL, aruco_dict, parameters=params)
    cornersR, idsR, rejectedImgPointsR = cv2.aruco.detectMarkers(grayR, aruco_dict, parameters=params)
    
    if len(cornersL) > 0 and len(cornersR) > 0:
        ic("# SUB PIXEL DETECTION LEFT")
        # SUB PIXEL DETECTION LEFT
        for corner in cornersL:
            cv2.cornerSubPix(grayL, corner,
                                winSize = (3,3),
                                zeroZone = (-1,-1),
                                criteria = criteria)
        res2L = cv2.aruco.interpolateCornersCharuco(cornersL,idsL,grayL,board)
        if res2L[1] is not None and res2L[2] is not None and len(res2L[1])>3 and decimatorL%1==0:
            allCornersL.append(res2L[1])
            allIdsL.append(res2L[2])

        # SUB PIXEL DETECTION RIGHT
        for corner in cornersR:
            cv2.cornerSubPix(grayR, corner,
                                winSize = (3,3),
                                zeroZone = (-1,-1),
                                criteria = criteria)
        res2R = cv2.aruco.interpolateCornersCharuco(cornersR,idsR,grayR,board)
        if res2R[1] is not None and res2R[2] is not None and len(res2R[1])>3 and decimatorR%1==0:
            allCornersR.append(res2R[1])
            allIdsR.append(res2R[2])

    decimatorL+=1
    decimatorR+=1

retvalL, camera_matrixL, dist_coeffsL, rvecsL, tvecsL = cv2.aruco.calibrateCameraCharuco(allCornersL, allIdsL, board, frameSize, None, None)
retvalR, camera_matrixR, dist_coeffsR, rvecsR, tvecsR = cv2.aruco.calibrateCameraCharuco(allCornersR, allIdsR, board, frameSize, None, None)

# Save calibration data
np.save('calibration_data/camera_matrixL.npy', camera_matrixL)
np.save('calibration_data/dist_coeffsR.npy', dist_coeffsL)
np.save('calibration_data/camera_matrixR.npy', camera_matrixR)
np.save('calibration_data/dist_coeffsR.npy', dist_coeffsR)


# shows undistorted image left
image_l = cv2.imread("images/test/left/test.jpg")
undistorted_image_l = cv2.undistort(image_l, camera_matrixL, dist_coeffsL)

# shows undistorted image right
image_r = cv2.imread("images/test/right/test.jpg")
undistorted_image_r = cv2.undistort(image_r, camera_matrixR, dist_coeffsR)

# plot distorted and undistorted images
fig, ax = plt.subplots(2, 2, figsize=(10, 5))
ax[0, 0].imshow(image_l)
ax[0, 0].set_title("Distorted Image Left")
ax[0, 0].axis('off')
ax[0, 1].imshow(undistorted_image_l)
ax[0, 1].set_title("Undistorted Image Left")
ax[0, 1].axis('off')
ax[1, 0].imshow(image_r)
ax[1, 0].set_title("Distorted Image Right")
ax[1, 0].axis('off')
ax[1, 1].imshow(undistorted_image_r)
ax[1, 1].set_title("Undistorted Image Right")
ax[1, 1].axis('off')
plt.show()

'''
#make gif
image_files = glob.glob('/Users/antonia/Desktop/gif_L/*.jpg')
counter = 0
for image_file in image_files: 
    image = cv2.imread(image_file)
    undistorted_image =cv2.undistort(image, camera_matrixL, dist_coeffsL)
    cv2.imwrite('/Users/antonia/Desktop/gif_undistorted/' + str(counter) + '.jpg', undistorted_image)
    counter += 1

'''

#allCorners,allIds,imsize


