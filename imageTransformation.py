# Script to resize all images to 16x16 to suit our model

import numpy as np
import cv2 
import csv 
import matplotlib.pyplot as plt

import os
import cv2

# Directory where your image files are located
image_dir = '\Data'

# Get a list of all files in the directory
all_files = os.listdir(image_dir)

# Filter files that start with "sw" and end with an image extension (e.g., jpg, png)
sw_files = [file for file in all_files if file.startswith('sw') and (file.endswith('.jpg') or file.endswith('.png'))]

# Loop through the filtered files and read each image using cv2
i=225

white_color = np.array([255, 255, 255])

for file_name in sw_files:
    file_path = os.path.join(image_dir, file_name)
    image = cv2.imread(file_path,cv2.IMREAD_UNCHANGED)

    if image is not None:

        # Do whatever you want with the image here
        # For example, you can display it using cv2.imshow or perform some processing

        image = cv2.resize(image, (16, 16))

        # Create an alpha channel with all non-white pixels set to 255 (fully opaque)
        alpha_channel = np.all(image[:, :, :3] != white_color, axis=2).astype(np.uint8) * 255
        # Add the alpha channel to the image
        if image.shape[2] == 3:
            image = np.dstack((image, alpha_channel))

        clr1 = np.array(image[1, 1][:3])
        clr2 = np.array(image[0, 0][:3])

        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                if np.array_equal(image[y, x][:3], clr2):  # Check if pixel is white
                    image[y, x] = [255,255,255,255]  # Set alpha channel to 0
                if np.array_equal(image[y, x][:3], clr1):  # Check if pixel is white
                    image[y, x] = [255,255,255,255]  # Set alpha channel to 0
                if all(image[y, x] == [0,0,0,0]):  # Check if pixel is white
                    image[y, x] = [255,255,255,255]  # Set alpha channel to 0
        cv2.imwrite("\Data\data16\csw"+str(i)+".png", image)
        i=i+1

# Wait for a key press and then close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()