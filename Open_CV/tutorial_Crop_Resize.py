#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 19:17:24 2021

@author: tai
"""

import cv2

path = "Resources/lena.png"
img = cv2.imread(path)

"""
In OpenCV,
- Coordinate: (Width, Height) not (Height, Width) in numpy
(0,0) at the Top-Left of the image
"""

print(type(img)) # <class 'numpy.ndarray'>
cv2.imshow("Img", img)
print(img.shape) # (512, 512, 3)

width, height = 400, 400
imgResize = cv2.resize(img, (width, height))
cv2.imshow("Img Resize Smaller", imgResize)
print(imgResize.shape) # (400, 400, 3)

width, height = 1000, 1000
imgResize2 = cv2.resize(img, (width, height))
cv2.imshow("Img Resize Bigger", imgResize2)
print(imgResize2.shape) # (1000, 1000, 3)


# Crop
"""
crop the sky and the road, to detect the lane.
"""
# Method: use numpy slide, to crop the middle lane
'''
NOTE: numpy use (height, width)
'''
imgCropped = img[0:300, 200:400]
cv2.imshow("Img Resize Smaller", imgCropped)
print(imgCropped.shape) # (300, 200, 3)

imgCropResize = cv2.resize(imgCropped, (img.shape[1], img.shape[0]))
cv2.imshow("Img Crop Resize", imgCropResize)



cv2.waitKey(0)
cv2.destroyAllWindows()