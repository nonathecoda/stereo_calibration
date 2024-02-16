from icecream import ic
import cv2
import numpy as np
import cv2, PIL, os
from cv2 import aruco
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
import glob


def draw_results(imgPoints_int, img):
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    for point in imgPoints_int:
        point = (int(point[0, 0]), int(point[0, 1]))
        cv2.circle(img, point, radius=5, color=(0, 255, 0), thickness=-1) # GREEN
    
    cv2.imshow('Image with Points', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def draw_epilines(img1, img2, pts1, pts2, F):
    """
    Draw epipolar lines on both images given corresponding points and the fundamental matrix.

    Parameters:
    - img1, img2: The two images between which the epipolar lines will be drawn.
    - pts1, pts2: Corresponding points in img1 and img2. Both should be Nx2 numpy arrays.
    - F: The fundamental matrix relating img1 and img2.
    """
    # Convert points to homogeneous coordinates
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines1 = lines1.reshape(-1, 3)
    lines2 = lines2.reshape(-1, 3)

    # Draw the lines on the images
    r, c = img1.shape[:2]
    for r, pt1, pt2 in zip(lines1, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
    r, c = img2.shape[:2]
    for r, pt1, pt2 in zip(lines2, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img2 = cv2.line(img2, (x0, y0), (x1, y1), color, 1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)

    # Display the images with epipolar lines
    plt.figure(figsize=(15, 5))
    plt.subplot(121), plt.imshow(img1)
    plt.title('Image 1'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img2)
    plt.title('Image 2'), plt.xticks([]), plt.yticks([])

    plt.show()

# ------------------------------
#https://medium.com/@ed.twomey1/using-charuco-boards-in-opencv-237d8bc9e40d
# ENTER YOUR PARAMETERS HERE:
'''
ARUCO_DICT = cv2.aruco.DICT_4X4_100
SQUARES_VERTICALLY = 12
SQUARES_HORIZONTALLY = 8
SQUARE_LENGTH = 67.5
MARKER_LENGTH = 50
LENGTH_PX = int(SQUARE_LENGTH*SQUARES_VERTICALLY)  # total length of the page in pixels
MARGIN_PX = 0    # size of the margin in pixels
# ------------------------------

ARUCO_DICT = cv2.aruco.DICT_5X5_250
SQUARES_VERTICALLY = 17
SQUARES_HORIZONTALLY = 11
SQUARE_LENGTH = 72
MARKER_LENGTH = 54
LENGTH_PX = int(SQUARE_LENGTH*SQUARES_VERTICALLY)  # total length of the page in pixels
MARGIN_PX = 0    # size of the margin in pixels


# ------------------------------
ARUCO_DICT = cv2.aruco.DICT_5X5_250
SQUARES_VERTICALLY = 17
SQUARES_HORIZONTALLY = 11
SQUARE_LENGTH = 100
MARKER_LENGTH = 75
LENGTH_PX = int(SQUARE_LENGTH*SQUARES_VERTICALLY)  # total length of the page in pixels
MARGIN_PX = 0    # size of the margin in pixels
'''
# ------------------------------
ARUCO_DICT = cv2.aruco.DICT_4X4_250
SQUARES_VERTICALLY = 8
SQUARES_HORIZONTALLY = 5
SQUARE_LENGTH = 200
MARKER_LENGTH = 150
LENGTH_PX = int(SQUARE_LENGTH*SQUARES_VERTICALLY)  # total length of the page in pixels
MARGIN_PX = 0    # size of the margin in pixels

SAVE_NAME = '/Users/antonia/dev/masterthesis/stereo_calibration/ChArUco_Marker.png'

aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
board = cv2.aruco.CharucoBoard((SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), SQUARE_LENGTH, MARKER_LENGTH, aruco_dict)
board.setLegacyPattern(True)
size_ratio = SQUARES_HORIZONTALLY / SQUARES_VERTICALLY
img = cv2.aruco.CharucoBoard.generateImage(board, (LENGTH_PX, int(LENGTH_PX*size_ratio)), marginSize=MARGIN_PX)

allCornersL = []
allIdsL = []
allCornersR = []
allIdsR = []
objpoints = [] # 3d point in real world space

stereoCornersR = []
stereoCornersL = []


frameSize = (3280,2464)

# SUB PIXEL CORNER DETECTION CRITERION
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

imagesLeft = sorted(glob.glob('images/charuco_4x4_large/left/*.jpg'))
imagesRight = sorted(glob.glob('images/charuco_4x4_large/right/*.jpg'))

for imgLeft, imgRight in zip(imagesLeft, imagesRight):
    #print(imgLeft, imgRight)
    imgL = cv2.imread(imgLeft)
    imgR = cv2.imread(imgRight)
   
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    frameSize = grayL.shape
    
    params = cv2.aruco.DetectorParameters()
    cornersL, idsL, rejectedImgPointsL = cv2.aruco.detectMarkers(grayL, aruco_dict, parameters=params)
    cornersR, idsR, rejectedImgPointsR = cv2.aruco.detectMarkers(grayR, aruco_dict, parameters=params)
    
    #if len(idsL) > 0 and len(idsR) > 0:
    if idsL is not None and idsR is not None:
        ################## LEFT ##################
        for corner in cornersL:
            cv2.cornerSubPix(grayL, corner,
                                winSize = (3,3),
                                zeroZone = (-1,-1),
                                criteria = criteria)
        res2L = cv2.aruco.interpolateCornersCharuco(cornersL,idsL,grayL,board)
        if res2L[1] is None:
            continue
        #draw_results(res2L[1], grayL)
        #if len(res2L[1]) > 5:
        ##    allCornersL.append(res2L[1])
        #    allIdsL.append(res2L[2])
        ################## RIGHT ##################
        for corner in cornersR:
            cv2.cornerSubPix(grayR, corner,
                                winSize = (3,3),
                                zeroZone = (-1,-1),
                                criteria = criteria)
        res2R = cv2.aruco.interpolateCornersCharuco(cornersR,idsR,grayR,board)
        if res2R[1] is None:
            continue
        #draw_results(res2R[1], grayR)
        #if len(res2R[1]) > 5:
            #allCornersR.append(res2R[1])
            #allIdsR.append(res2R[2])
        #ic(res2L[1], res2R[1])
        
        ################## TOTAL ##################
        '''
        if len(res2L[1]) == len(res2R[1]) and len(res2L[1]) > 3:
            objpoints_L, imgpoints_L = cv2.aruco.CharucoBoard.matchImagePoints(board, res2L[1], res2L[2])
            stereoCornersR.append(res2R[1])
            stereoCornersL.append(res2L[1])
            objpoints.append(objpoints_L)
        '''
        
        if len(res2L[1]) == len(res2R[1]) and len(res2L[1]) > 3:
            objpoints_L, imgpoints_L = cv2.aruco.CharucoBoard.matchImagePoints(board, res2L[1], res2L[2])
            objpoints_R, imgpoints_R = cv2.aruco.CharucoBoard.matchImagePoints(board, res2R[1], res2R[2])
            allCornersL.append(res2L[1])
            allIdsL.append(res2L[2])
            objpoints.append(objpoints_L)
            allCornersR.append(res2R[1])
            allIdsR.append(res2R[2])
            
        


focal_length = 0.3528 # in pixels

retvalL, camera_matrixL, dist_coeffsL, rvecsL, tvecsL = cv2.aruco.calibrateCameraCharuco(allCornersL, allIdsL, board, frameSize, None, None)
retvalR, camera_matrixR, dist_coeffsR, rvecsR, tvecsR = cv2.aruco.calibrateCameraCharuco(allCornersR, allIdsR, board, frameSize, None, None)

#retvalL, camera_matrixL, dist_coeffsL, rvecsL, tvecsL = cv2.fisheye.calibrate(objpoints, allCornersL, grayL.shape[::-1], K = camera_matrixL, D = dist_coeffsL)
#retvalR, camera_matrixR, dist_coeffsR, rvecsR, tvecsR = cv2.fisheye.calibrate(objpoints, allCornersR, grayR.shape[::-1], K = camera_matrixR, D = dist_coeffsR)


ic(rvecsL)

ic(retvalL)
ic(retvalR)

ic(camera_matrixL)
ic(camera_matrixR)

# Save calibration data
np.save('calibration_data/camera_matrixL.npy', camera_matrixL)
np.save('calibration_data/dist_coeffsL.npy', dist_coeffsL)
np.save('calibration_data/camera_matrixR.npy', camera_matrixR)
np.save('calibration_data/dist_coeffsR.npy', dist_coeffsR)

# shows undistorted image left
image_l = cv2.imread("images/test/left/test.jpg")
undistorted_image_l = cv2.undistort(image_l, camera_matrixL, dist_coeffsL)


image_r = cv2.imread("images/test/right/test.jpg")
undistorted_image_r = cv2.undistort(image_r, camera_matrixR, dist_coeffsR)
cv2.imshow("undistorted image left", undistorted_image_l)
cv2.waitKey(0)
cv2.destroyAllWindows()

########## Stereo Vision Calibration #############################################

flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC
# Here we fix the intrinsic camara matrixes so that only Rot, Trns, Emat and Fmat are calculated.
# Hence intrinsic parameters are the same 

criteria_stereo= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)

# This step is performed to transformation between the two cameras and calculate Essential and Fundamenatl matrix
retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix = cv2.stereoCalibrate(objpoints, stereoCornersL, stereoCornersR, camera_matrixL, dist_coeffsL, camera_matrixR, dist_coeffsR, frameSize, criteria_stereo, flags)
#retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix = cv2.stereoCalibrate(objpoints, stereoCornersL, stereoCornersL, camera_matrixL, dist_coeffsL, camera_matrixL, dist_coeffsL, frameSize, criteria_stereo, flags)
ic(trans)
ic(retStereo)
########## Stereo Rectification #################################################

rectifyScale= 0
rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R= cv2.stereoRectify(newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], rot, trans, rectifyScale,(0,0))

stereoMapL = cv2.initUndistortRectifyMap(newCameraMatrixL, distL, rectL, projMatrixL, grayL.shape[::-1], cv2.CV_16SC2)
stereoMapR = cv2.initUndistortRectifyMap(newCameraMatrixR, distR, rectR, projMatrixR, grayR.shape[::-1], cv2.CV_16SC2)

file_stereo_map = 'calibration_data/stereoMap1.xml'
cv_file = cv2.FileStorage(file_stereo_map, cv2.FILE_STORAGE_WRITE)
print("Saving parameters to " + file_stereo_map + "!")

cv_file.write('stereoMapL_x',stereoMapL[0])
cv_file.write('stereoMapL_y',stereoMapL[1])
cv_file.write('stereoMapR_x',stereoMapR[0])
cv_file.write('stereoMapR_y',stereoMapR[1])

cv_file.release()