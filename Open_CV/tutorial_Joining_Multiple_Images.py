#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 21:19:47 2021

@author: tai
"""

import cv2
import numpy as np

path = "Resources/lena.png"
img = cv2.imread(path)
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 0)
imgCanny = cv2.Canny(imgBlur, 100, 200)

kernel = np.ones((5, 5), dtype=np.uint8)
imgDilation = cv2.dilate(imgCanny, kernel, iterations = 2)
imgEroded = cv2.erode(imgDilation, kernel, iterations = 2)



####################### Horizontal and Vertical stack Image without using function #########################
hor = np.hstack((img, img))
ver = np.vstack((img, img))

cv2.imshow("Horizontal Images", hor)
cv2.imshow("Vertical Images", ver)
'''
Problem 1: 2 images must in the same size *****************************
Problem 2: 2 images must have the same channels **********************
mean that we cannot stack Gray image and BGR images together.
'''

from Stack_Image import stackImages 

StackedImages = stackImages(([img, imgGray, imgBlur], [imgCanny, imgDilation, imgEroded]), 0.8)

imgBlank = np.zeros((200, 200), dtype = np.uint8)
# stackImages resize all image to the first img (TOP - LEFT image)
#StackedImages = stackImages(([img, imgGray, imgBlur], [imgCanny, imgDilation, imgBlank]), 0.8)

cv2.imshow("Stacked Images",StackedImages)

cv2.waitKey(0)
cv2.destroyAllWindows()

####################### Webcam Stack Images #######################
import cv2

frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)

while True:
    success, img = cap.read()
    
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 0)
    imgCanny = cv2.Canny(imgBlur, 100, 200)
    
    kernel = np.ones((5, 5), dtype=np.uint8)
    imgDilation = cv2.dilate(imgCanny, kernel, iterations = 2)
    imgEroded = cv2.erode(imgDilation, kernel, iterations = 2)
    
    from Stack_Image import stackImages 
    '''
    in folder
    from Other_folder import Stack_Images
    Stack_Images.stackImages()
    '''
    #StackedImages = stackImages(([img, imgGray, imgBlur], [imgCanny, imgDilation, imgEroded]), 0.8)
    ############# ERROR : scale NOT WORKING **************************************
    StackedImages = stackImages(([img, imgGray, imgBlur, imgCanny, imgDilation, imgEroded]
                                 , [img, imgGray, imgBlur, imgCanny, imgDilation, imgEroded]), scale=0.1)
    cv2.imshow("Stacked Images",StackedImages)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
#cv2.waitKey(0)
cv2.destroyAllWindows()


