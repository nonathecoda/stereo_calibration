import numpy as np
import cv2 as cv
from icecream import ic
import os

img_folder = "images/charuco_5x5_art_people/before_split/"
output_folder ="images/charuco_5x5_art_people/"

#### delete previews ####
# Loop over each file in the directory
for filename in os.listdir(img_folder):
    # Construct the full file path
    file_path = os.path.join(img_folder, filename)
    
    # Check if the file contains "preview" in its name
    if "preview" in filename:
        # Ensure that it is a file, not a directory or symbolic link
        if os.path.isfile(file_path):
            # Delete the file
            os.remove(file_path)
            print(f"Deleted: {file_path}")

### split remaining images ###
for subdir, dirs, files in os.walk(img_folder):
    for file in files:
        img = cv.imread(os.path.join(subdir, file), 1)
        ic(os.path.join(subdir, file))
        if img is None:
            continue
        img_l = img[:, :int(img.shape[1]/2), :]
        img_r = img[:, int(img.shape[1]/2):, :]
        cv.imwrite(output_folder + "left/" + file, img_l)
        cv.imwrite(output_folder + "right/" + file, img_r)