from icecream import ic
import cv2
import numpy as np
import cv2, PIL, os
from cv2 import aruco
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
import glob

''''
Using all the charuco boards to calibrate the camera and the stereo camera

'''
def draw_results(imgPoints_int, img, side='left'):
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    for point in imgPoints_int:
        point = (int(point[0, 0]), int(point[0, 1]))
        cv2.circle(img, point, radius=5, color=(0, 255, 0), thickness=-1) # GREEN
    
    cv2.imshow(str(side + ' image'), img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def draw_stereo_results(imgL, imgR, res2L, res2R):
    img = np.hstack((imgL, imgR))
    # move res2R[1] to the right
    res2R[1][:, :, 0] += imgL.shape[1]
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    for point in res2L[1]:
        point = (int(point[0, 0]), int(point[0, 1]))
        cv2.circle(img, point, radius=5, color=(0, 255, 0), thickness=-1)
    for point in res2R[1]:
        point = (int(point[0, 0]), int(point[0, 1]))
        cv2.circle(img, point, radius=5, color=(0, 255, 0), thickness=-1)
    cv2.imshow('Stereo Result', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def create_charuco_board(ARUCO_DICT, SQUARES_VERTICALLY, SQUARES_HORIZONTALLY, SQUARE_LENGTH, MARKER_LENGTH):
    
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard((SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), SQUARE_LENGTH, MARKER_LENGTH, aruco_dict)
    board.setLegacyPattern(True)

    return board, aruco_dict

board_image_array = []

board_charuco_5x5, aru_dict = create_charuco_board(cv2.aruco.DICT_5X5_250, 17, 11, 100, 75)
imagesLeft_charuco_5x5_pure = sorted(glob.glob('images/charuco_5x5_pure/left/*.jpg'))
imagesRight_charuco_5x5_pure = sorted(glob.glob('images/charuco_5x5_pure/right/*.jpg'))
#board_image_array.append([imagesLeft_charuco_5x5_pure, imagesRight_charuco_5x5_pure, board_charuco_5x5, aru_dict])

imagesLeft_charuco_5x5_pablo = sorted(glob.glob('images/charuco_5x5_pablo/left/*.jpg'))
imagesRight_charuco_5x5_pablo = sorted(glob.glob('images/charuco_5x5_pablo/right/*.jpg'))
#board_image_array.append([imagesLeft_charuco_5x5_pablo, imagesRight_charuco_5x5_pablo, board_charuco_5x5, aru_dict])

board_4x4_sticker, aru_dict = create_charuco_board(cv2.aruco.DICT_4X4_250, 8, 5, 200, 150)
imagesLeft_4x4_large = sorted(glob.glob('images/charuco_4x4_large/left/*.jpg'))
imagesRight_4x4_large = sorted(glob.glob('images/charuco_4x4_large/right/*.jpg'))
board_image_array.append([imagesLeft_4x4_large, imagesRight_4x4_large, board_4x4_sticker, aru_dict])

board_charuco_5x5_screen, aru_dict = create_charuco_board(cv2.aruco.DICT_5X5_250, 17, 11, 72, 54)
imagesLeft_charuco_5x5_72_54 = sorted(glob.glob('images/charuco_5x5_72_54/left/*.jpg'))
imagesRight_charuco_5x5_72_54 = sorted(glob.glob('images/charuco_5x5_72_54/right/*.jpg'))
#board_image_array.append([imagesLeft_charuco_5x5_72_54, imagesRight_charuco_5x5_72_54, board_charuco_5x5_screen, aru_dict])

###########LOS GEHTS##################


objpoints = [] # 3d point in real world space

stereoCornersR = []
stereoCornersL = []


frameSize = (3280,2464)

# SUB PIXEL CORNER DETECTION CRITERION
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

focal_length = 0.3528
camera_matrixL = np.array([[focal_length, 0,              frameSize[0]/2],
                                [0,            focal_length,   frameSize[1]/2],
                                [0,            0,              1]])
camera_matrixR = camera_matrixL.copy()

for board_img in board_image_array:
    allCornersL = []
    allIdsL = []
    allCornersR = []
    allIdsR = []
    aru_dict = board_img[3]
    board = board_img[2]
    for imgLeft, imgRight in zip(board_img[0], board_img[1]):

        #print(imgLeft, imgRight)
        imgL = cv2.imread(imgLeft)
        imgR = cv2.imread(imgRight)
    
        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
        frameSize = grayL.shape
        
        params = cv2.aruco.DetectorParameters()
        cornersL, idsL, rejectedImgPointsL = cv2.aruco.detectMarkers(grayL, aru_dict, parameters=params)
        cornersR, idsR, rejectedImgPointsR = cv2.aruco.detectMarkers(grayR, aru_dict, parameters=params)
        
        #if len(idsL) > 0 and len(idsR) > 0:
        if idsL is not None and idsR is not None:
            ################## RIGHT ##################
            for corner in cornersR:
                cv2.cornerSubPix(grayR, corner,
                                    winSize = (3,3),
                                    zeroZone = (-1,-1),
                                    criteria = criteria)
            res2R = cv2.aruco.interpolateCornersCharuco(cornersR,idsR,grayR,board)
            if res2R[1] is None:
                continue
            #draw_results(res2R[1], grayR, 'right')
            if len(res2R[1]) > 5:
                allCornersR.append(res2R[1])
                allIdsR.append(res2R[2])
            ################## LEFT ##################
            for corner in cornersL:
                cv2.cornerSubPix(grayL, corner,
                                    winSize = (3,3),
                                    zeroZone = (-1,-1),
                                    criteria = criteria)
            res2L = cv2.aruco.interpolateCornersCharuco(cornersL,idsL,grayL,board)
            if res2L[1] is None:
                continue
            #draw_results(res2L[1], grayL, 'left')
            if len(res2L[1]) > 5:
                allCornersL.append(res2L[1])
                allIdsL.append(res2L[2])
            
    # use initial guess for intrinsic parameters
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_RATIONAL_MODEL)

    retvalL, camera_matrixL, dist_coeffsL, rvecsL, tvecsL = cv2.aruco.calibrateCameraCharuco(allCornersL, allIdsL, board, frameSize, camera_matrixL, None)
    retvalR, camera_matrixR, dist_coeffsR, rvecsR, tvecsR = cv2.aruco.calibrateCameraCharuco(allCornersR, allIdsR, board, frameSize, camera_matrixR, None)

ic(retvalL)
ic(retvalR)

ic(camera_matrixL)
ic(camera_matrixR)

# Save calibration data
np.save('calibration_data/camera_matrixL.npy', camera_matrixL)
np.save('calibration_data/dist_coeffsL.npy', dist_coeffsL)
np.save('calibration_data/camera_matrixR.npy', camera_matrixR)
np.save('calibration_data/dist_coeffsR.npy', dist_coeffsR)

# shows undistorted example images
image_l = cv2.imread("images/test/left/test.jpg")
undistorted_image_l = cv2.undistort(image_l, camera_matrixL, dist_coeffsL)

image_r = cv2.imread("images/test/right/test.jpg")
undistorted_image_r = cv2.undistort(image_r, camera_matrixR, dist_coeffsR)
cv2.imshow("undistorted image left", undistorted_image_l)
cv2.waitKey(0)
cv2.destroyAllWindows()


exit()
########## Stereo Vision Calibration #############################################


######################################################
### charuco board for stereo camera calibration
######################################################
ARUCO_DICT_stereo = cv2.aruco.DICT_4X4_250
SQUARES_VERTICALLY_stereo = 8
SQUARES_HORIZONTALLY_stereo = 5
SQUARE_LENGTH_stereo = 200
MARKER_LENGTH_stereo = 150
LENGTH_PX_stereo = int(SQUARE_LENGTH_stereo*SQUARES_VERTICALLY_stereo)  # total length of the page in pixels
MARGIN_PX_stereo = 0    # size of the margin in pixels

aruco_dict_stereo = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_stereo)
board_stereo = cv2.aruco.CharucoBoard((SQUARES_VERTICALLY_stereo, SQUARES_HORIZONTALLY_stereo), SQUARE_LENGTH_stereo, MARKER_LENGTH_stereo, aruco_dict_stereo)
board_stereo.setLegacyPattern(True)
size_ratio_stereo = SQUARES_HORIZONTALLY_single / SQUARES_VERTICALLY_single
img_stereo = cv2.aruco.CharucoBoard.generateImage(board_stereo, (LENGTH_PX_single, int(LENGTH_PX_single*size_ratio_stereo)), marginSize=MARGIN_PX_single)

### get obj points for stereo calibration

################## TOTAL ##################

for imgLeft, imgRight in zip(imagesLeft_stereo, imagesRight_stereo):
    
    imgL = cv2.imread(imgLeft)
    imgR = cv2.imread(imgRight)
   
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    
    frameSize = grayL.shape
    
    params = cv2.aruco.DetectorParameters()
    cornersL, idsL, rejectedImgPointsL = cv2.aruco.detectMarkers(grayL, aruco_dict_stereo, parameters=params)
    cornersR, idsR, rejectedImgPointsR = cv2.aruco.detectMarkers(grayR, aruco_dict_stereo, parameters=params)
    
    #if len(idsL) > 0 and len(idsR) > 0:
    if idsL is not None and idsR is not None:
        ################## LEFT ##################
        for corner in cornersL:
            cv2.cornerSubPix(grayL, corner,
                                winSize = (3,3),
                                zeroZone = (-1,-1),
                                criteria = criteria)
        res2L = cv2.aruco.interpolateCornersCharuco(cornersL,idsL,grayL,board_stereo)
        if res2L[1] is None:
            continue
        #draw_results(res2L[1], grayL)
        
        ################## RIGHT ##################
        for corner in cornersR:
            cv2.cornerSubPix(grayR, corner,
                                winSize = (3,3),
                                zeroZone = (-1,-1),
                                criteria = criteria)
        res2R = cv2.aruco.interpolateCornersCharuco(cornersR,idsR,grayR,board_stereo)
        if res2R[1] is None:
            continue
        #draw_results(res2R[1], grayR)
        
        ################## TOTAL ##################
        if len(res2L[1]) == len(res2R[1]) and len(res2L[1]) > 3:
            objpoints_L, imgpoints_L = cv2.aruco.CharucoBoard.matchImagePoints(board_stereo, res2L[1], res2L[2])
            stereoCornersR.append(res2R[1])
            stereoCornersL.append(res2L[1])
            objpoints.append(objpoints_L)
            #draw_stereo_results(grayL, grayR, res2L, res2R)


flags = 0
#flags |= cv2.CALIB_FIX_INTRINSIC
# Here we fix the intrinsic camara matrixes so that only Rot, Trns, Emat and Fmat are calculated.
# Hence intrinsic parameters are the same 

criteria_stereo= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)

# This step is performed to transformation between the two cameras and calculate Essential and Fundamenatl matrix
retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix = cv2.stereoCalibrate(objpoints, stereoCornersL, stereoCornersR, camera_matrixL, dist_coeffsL, camera_matrixR, dist_coeffsR, frameSize, criteria_stereo, flags)
#retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix = cv2.stereoCalibrate(objpoints, stereoCornersL, stereoCornersL, camera_matrixL, dist_coeffsL, camera_matrixL, dist_coeffsL, frameSize, criteria_stereo, flags)
ic(newCameraMatrixL)
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