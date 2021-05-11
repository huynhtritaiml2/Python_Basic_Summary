#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 18:48:36 2021

@author: tai
"""

import cv2
import numpy as np

path = "Resources/lena.png"
img = cv2.imread(path)
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imglur = cv2.GaussianBlur(imgGray, (7,7), 0) # Kernel in Odd number, increse the size more blur
imgCanny = cv2.Canny(imgGray, 100, 100)
imgCanny2 = cv2.Canny(imgGray, 50, 50) # more edge
imgCanny3 = cv2.Canny(imgGray, 50, 200)

# Dilation: sự giãn nở 
kernel = np.ones((5, 5), np.uint8)
imgDilation = cv2.dilate(imgCanny3, kernel, iterations = 1)

imgEroded = cv2.erode(imgDilation, kernel, iterations = 1)

from Stack_Image import stackImages as stackImages


StackImages = stackImages(([img, imgGray, imglur, imgCanny3]
                           , [imgDilation, imgEroded, img, img]
                           ), 0.2)

cv2.imshow("Stack Image", StackImages)
#cv2.imshow("Lena", img)
cv2.waitKey(0)
cv2.destroyAllWindows()