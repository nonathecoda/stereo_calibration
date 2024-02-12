from icecream import ic
import cv2
import numpy as np
import cv2, PIL, os
from cv2 import aruco
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
import glob


def draw_results(imgPoints_int, imgPoints_matchimgPoints, objPoints, img):
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    
    for point in imgPoints_matchimgPoints:
        point = (int(point[0, 0]), int(point[0, 1]))
        ic(point)
        cv2.circle(img, point, radius=5, color=(255, 0, 0), thickness=-1) # BLUE
    for point in imgPoints_int:
        point = (int(point[0, 0]), int(point[0, 1]))
        ic(point)
        cv2.circle(img, point, radius=5, color=(0, 255, 0), thickness=-1) # GREEN
    for point in objPoints:
        point = (int(point[0, 0]), int(point[0, 1]))
        ic(point)
        cv2.circle(img, point, radius=5, color=(0, 0, 255), thickness=-1)    # RED
    
    ic(img.shape)
    cv2.imshow('Image with Points', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    exit()

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
objpoints = [] # 3d point in real world space

objp = np.zeros(((SQUARES_VERTICALLY-1) * (SQUARES_HORIZONTALLY-1), 3), np.float32)
objp[:,:2] = np.mgrid[0:(SQUARES_VERTICALLY-1),0:(SQUARES_HORIZONTALLY-1)].T.reshape(-1,2)

objp = objp * SQUARE_LENGTH # should this be in mm? right now its in meters
ic(objp.shape)
#exit()
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
    
        ################## LEFT ##################
        for corner in cornersL:
            cv2.cornerSubPix(grayL, corner,
                                winSize = (3,3),
                                zeroZone = (-1,-1),
                                criteria = criteria)
        res2L = cv2.aruco.interpolateCornersCharuco(cornersL,idsL,grayL,board)
        objpoints_L, imgpoints_L = cv2.aruco.CharucoBoard.matchImagePoints(board, res2L[1], res2L[2])
        #objpoints_L, imgpoints_L = cv2.aruco.getBoardObjectAndImagePoints(board, res2L[1], res2L[2])

        ################## RIGHT ##################
        for corner in cornersR:
            cv2.cornerSubPix(grayR, corner,
                                winSize = (3,3),
                                zeroZone = (-1,-1),
                                criteria = criteria)
        res2R = cv2.aruco.interpolateCornersCharuco(cornersR,idsR,grayR,board)
        objpoints_R, imgpoints_R = cv2.aruco.CharucoBoard.matchImagePoints(board, res2R[1], res2R[2])
        #objpoints_R, imgpoints_R = cv2.aruco.getBoardObjectAndImagePoints(board, res2R[1], res2R[2])
        # seems like imgpoints_R is the same as res2R[1]
        
        ################## TOTAL ##################
        if len(res2L[1]) == len(res2R[1]) and len(res2L[1]) > 3:
            allCornersL.append(res2L[1])
            allIdsL.append(res2L[2])
            objpoints.append(objpoints_L* SQUARE_LENGTH*100)
            allCornersR.append(res2R[1])
            allIdsR.append(res2R[2])
            #draw_results(res2L[1], imgpoints_L, objpoints_L, grayL)

    decimatorL+=1
    decimatorR+=1


retvalL, camera_matrixL, dist_coeffsL, rvecsL, tvecsL = cv2.aruco.calibrateCameraCharuco(allCornersL, allIdsL, board, frameSize, None, None)
#newCameraMatrixL, roi_L = cv2.getOptimalNewCameraMatrix(camera_matrixL, dist_coeffsL, frameSize, 0, frameSize)

retvalR, camera_matrixR, dist_coeffsR, rvecsR, tvecsR = cv2.aruco.calibrateCameraCharuco(allCornersR, allIdsR, board, frameSize, None, None)
#newCameraMatrixR, roi_R = cv2.getOptimalNewCameraMatrix(camera_matrixR, dist_coeffsR, frameSize, 0, frameSize)

# Save calibration data
np.save('calibration_data/camera_matrixL.npy', camera_matrixL)
np.save('calibration_data/dist_coeffsR.npy', dist_coeffsL)
np.save('calibration_data/camera_matrixR.npy', camera_matrixR)
np.save('calibration_data/dist_coeffsR.npy', dist_coeffsR)

'''
# shows undistorted image left
image_l = cv2.imread("images/test/left/test.jpg")
undistorted_image_l = cv2.undistort(image_l, camera_matrixL, dist_coeffsL)
cv2.imshow("undistorted image left", undistorted_image_l)
cv2.waitKey(0)
cv2.destroyAllWindows()
exit()

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


#make gif
image_files = glob.glob('/Users/antonia/Desktop/gif_L/*.jpg')
counter = 0
for image_file in image_files: 
    image = cv2.imread(image_file)
    undistorted_image =cv2.undistort(image, camera_matrixL, dist_coeffsL)
    cv2.imwrite('/Users/antonia/Desktop/gif_undistorted/' + str(counter) + '.jpg', undistorted_image)
    counter += 1

'''



########## Stereo Vision Calibration #############################################

flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC
# Here we fix the intrinsic camara matrixes so that only Rot, Trns, Emat and Fmat are calculated.
# Hence intrinsic parameters are the same 

criteria_stereo= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# This step is performed to transformation between the two cameras and calculate Essential and Fundamenatl matrix
ic(objpoints[0])
ic(allCornersL[0])
retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix = cv2.stereoCalibrate(objpoints, allCornersL, allCornersR, camera_matrixL, dist_coeffsL, camera_matrixR, dist_coeffsR, frameSize, criteria_stereo, flags)
ic(retStereo)
ic(trans)
ic(newCameraMatrixR)
########## Stereo Rectification #################################################

rectifyScale= 0
rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R= cv2.stereoRectify(newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], rot, trans, rectifyScale,(0,0))

stereoMapL = cv2.initUndistortRectifyMap(newCameraMatrixL, distL, rectL, projMatrixL, grayL.shape[::-1], cv2.CV_16SC2)
stereoMapR = cv2.initUndistortRectifyMap(newCameraMatrixR, distR, rectR, projMatrixR, grayR.shape[::-1], cv2.CV_16SC2)

print("Saving parameters!")
cv_file = cv2.FileStorage('calibration_data/stereoMap.xml', cv2.FILE_STORAGE_WRITE)

cv_file.write('stereoMapL_x',stereoMapL[0])
cv_file.write('stereoMapL_y',stereoMapL[1])
cv_file.write('stereoMapR_x',stereoMapR[0])
cv_file.write('stereoMapR_y',stereoMapR[1])

cv_file.release()