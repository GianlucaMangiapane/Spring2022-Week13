"""
Filename: build_dataset.py

Function: Crops the images into small 256x256 images and divides the dataset into training and testing set.

Author: Jerin Paul (https://github.com/Paulymorphous)
Website: https://www.livetheaiexperience.com/
"""

import numpy as np
import cv2
from tqdm import tqdm
import os
import math
import time
import matplotlib.pyplot as plt
      
       
root_data_path = "../Data/MassachusettsRoads_n50/"
images_path = root_data_path + "Train/Images/samples//"
masks_path = root_data_path + "Train/Masks/samples/"

image_file = '9_10378750_15.tiff'
image_file = '29_10378765_15.tiff'
image_file = '35_10378795_15.tiff'

## Read image and mask
image_path = images_path + image_file
image = cv2.imread(image_path)

mask_path = masks_path + image_file
mask = cv2.imread(mask_path, 0)


## Create a composite image
mask3 = np.zeros([256,256,3])
mask3[:,:,0] = mask
imageComb = np.maximum(image, mask3).astype(int)

## Show img
fig, axs = plt.subplots(1, 3, figsize=[12, 4])
axs[0].imshow(image)
axs[1].imshow(mask)
axs[2].imshow(imageComb)
plt.show()



