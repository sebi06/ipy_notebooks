#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 18:56:04 2020

@author: sebi06
"""

import cv2
import numpy as np
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
from aicsimageio import AICSImage, imread, imread_dask
import numpy as np
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from skimage.filters import threshold_otsu, threshold_triangle, rank
from skimage.segmentation import clear_border
from skimage.color import label2rgb
from skimage.util import invert
from skimage.filters import median
from skimage.morphology import closing, square
from skimage.morphology import remove_small_objects, remove_small_holes, disk
from skimage.segmentation import random_walker
from scipy import ndimage
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt



#filename = r'C:\Users\m1srh\Documents\Testdata_Zeiss\Castor\testwell96.czi'
filename = r'/datadisk1/tuxedo/testpictures/Testdata_Zeiss/wellplate/testwell96.czi'

# Get an AICSImage object
img = AICSImage(filename)


image2d = img.get_image_data("YX", S=0, T=0, Z=0, C=0)
image2d = image2d[700:900, 500:700]


thresh = cv2.threshold(image2d, binary, cv2.THRESH_OTSU)

# Compute Euclidean distance from every binary pixel
# to the nearest zero pixel then find peaks
distance_map = ndimage.distance_transform_edt(thresh)
local_max = peak_local_max(distance_map, indices=False, min_distance=20, labels=thresh)

# Perform connected component analysis then apply Watershed
markers = ndimage.label(local_max, structure=np.ones((3, 3)))[0]
labels = watershed(-distance_map, markers, mask=thresh)

# Iterate through unique labels
total_area = 0
for label in np.unique(labels):
    if label == 0:
        continue

    # Create a mask
    mask = np.zeros(image2d.shape, dtype="uint8")
    mask[labels == label] = 255

    # Find contours and determine contour area
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    total_area += area
    cv2.drawContours(image2d, [c], -1, (36,255,12), 4)

print(total_area)
cv2.imshow('image', image2d)
cv2.waitKey()

img.close()