from icecream import ic
import cv2 as cv2
import glob
import numpy as np
from matplotlib import pyplot as plt
import kornia as K
import kornia.feature as KF
import torch
#from kornia_moons.viz import draw_LAF_matches

imagesLeft = sorted(glob.glob('images/charuco_5x5_pure/left/*.jpg'))
imagesRight = sorted(glob.glob('images/charuco_5x5_pure/right/*.jpg'))

for imgLeft, imgRight in zip(imagesLeft, imagesRight):
    #print(imgLeft, imgRight)
    imgL_numpy = cv2.imread(imgLeft, 1)
    imgR_numpy = cv2.imread(imgRight, 1)

    # load camera matrix and distortion coefficients
    camera_matrixL = np.load('calibration_data/camera_matrixL.npy')
    dist_coeffsL = np.load('calibration_data/dist_coeffsL.npy')
    camera_matrixR = np.load('calibration_data/camera_matrixR.npy')
    dist_coeffsR = np.load('calibration_data/dist_coeffsR.npy')

    imgR_numpy = cv2.undistort(imgR_numpy, camera_matrixR, dist_coeffsR)
    imgL_numpy = cv2.undistort(imgL_numpy, camera_matrixL, dist_coeffsL)

    imgR = K.image_to_tensor(imgR_numpy)
    imgL = K.image_to_tensor(imgL_numpy)
    # add dimension to tensor
    imgR = imgR.unsqueeze(0)
    imgL = imgL.unsqueeze(0)

    ic(type(imgL))
    
    #imgL = K.io.load_image(imgLeft, K.io.ImageLoadType.RGB32)[None, ...]
    #imgR = K.io.load_image(imgRight, K.io.ImageLoadType.RGB32)[None, ...]
    #3xHxW / in torch.uint in range [0,255] in "cpu"


    imgL = K.geometry.resize(imgL, (600, 375), antialias=True)
    imgR = K.geometry.resize(imgR, (600, 375), antialias=True)


    matcher = KF.LoFTR(pretrained="outdoor")

    input_dict = {
        "image0": K.color.rgb_to_grayscale(imgL.float()),  # LofTR works on grayscale images only
        "image1": K.color.rgb_to_grayscale(imgR.float()),
    }

    
    with torch.inference_mode():
        correspondences = matcher(input_dict)
    
    mkpts1 = correspondences["keypoints0"].cpu().numpy()
    mkpts2 = correspondences["keypoints1"].cpu().numpy()
    Fm, inliers = cv2.findFundamentalMat(mkpts1, mkpts2, cv2.USAC_MAGSAC, 0.5, 0.999, 100000)
    inliers = inliers > 0
    
    pts1 = np.int32(mkpts1)
    pts2 = np.int32(mkpts2)
    fundamental_matrix, inliers = cv2.findFundamentalMat(mkpts1, mkpts2, cv2.FM_RANSAC)

    
    def drawlines(img1src, img2src, lines, pts1src, pts2src):
        ''' img1 - image on which we draw the epilines for the points in img2
            lines - corresponding epilines '''
        ic(img1src.shape)
        r, c = (img1src.shape[0], img1src.shape[1])
        img1color = img1src
        img2color = img2src
        # Edit: use the same random seed so that two images are comparable!
        np.random.seed(0)
        for r, pt1, pt2 in zip(lines, pts1src, pts2src):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            x0, y0 = map(int, [0, -r[2]/r[1]])
            x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
            img1color = cv2.line(img1color, (x0, y0), (x1, y1), color, 1)
            img1color = cv2.circle(img1color, tuple(pt1), 5, color, -1)
            img2color = cv2.circle(img2color, tuple(pt2), 5, color, -1)
        return img1color, img2color


    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv2.computeCorrespondEpilines(
        pts2.reshape(-1, 1, 2), 2, fundamental_matrix)
    lines1 = lines1.reshape(-1, 3)
    img5, img6 = drawlines(imgL_numpy, imgR_numpy, lines1, pts1, pts2)

    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(
        pts1.reshape(-1, 1, 2), 1, fundamental_matrix)
    lines2 = lines2.reshape(-1, 3)
    img3, img4 = drawlines(imgL_numpy, imgR_numpy, lines2, pts2, pts1)

    plt.subplot(121), plt.imshow(img5)
    plt.subplot(122), plt.imshow(img3)
    plt.suptitle("Epilines in both images")
    plt.show()
    
   
    h1, w1 = imgL.shape[0], imgL.shape[1]
    h2, w2 = h1, w1
    _, H1, H2 = cv2.stereoRectifyUncalibrated(
        np.float32(pts1), np.float32(pts2), fundamental_matrix, imgSize=(w1, h1)
    )
    img1_rectified = cv2.warpPerspective(imgL_numpy, H1, (w1, h1))
    img2_rectified = cv2.warpPerspective(imgR_numpy, H2, (w2, h2))

    fig, axes = plt.subplots(1, 2, figsize=(15, 10))
    axes[0].imshow(img1_rectified, cmap="gray")
    axes[1].imshow(img2_rectified, cmap="gray")
    axes[0].axhline(250)
    axes[1].axhline(250)
    axes[0].axhline(450)
    axes[1].axhline(450)
    #plt.suptitle("Rectified images")
    #plt.savefig("rectified_images.png")
    plt.show()
    
    exit()