#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 09:53:57 2021

@author: tai
"""

import cv2
import numpy as np 

frameWidth, frameHeight = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)


def empty(a):
    pass

cv2.namedWindow("HSV")
cv2.resizeWindow("HSV", 640, 240)
# cv2.createTrackBar(name_string, name_window, min_value, max_value, function)
cv2.createTrackbar("HUE min", "HSV", 0, 179, empty) # Hue in HSV from 0->360-1 but OpenCV is from 0->180-1
cv2.createTrackbar("HUE max", "HSV", 17, 179, empty)
cv2.createTrackbar("SAT min", "HSV", 229, 255, empty)
cv2.createTrackbar("SAT max", "HSV", 255, 255, empty)
cv2.createTrackbar("VAL min", "HSV", 75, 255, empty)
cv2.createTrackbar("VAL max", "HSV", 255, 255, empty)
'''
https://stackoverflow.com/questions/48109650/how-to-detect-two-different-colors-using-cv2-inrange-in-python-opencv
Color HSV range detection
'''
while True:
    success, img = cap.read()
    '''
    hue: color
    saturation: how pure the color is
    value: how bright the color is
    '''
    imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    h_min = cv2.getTrackbarPos("HUE min", "HSV")
    h_max = cv2.getTrackbarPos("HUE max", "HSV")
    s_min = cv2.getTrackbarPos("SAT min", "HSV")
    s_max = cv2.getTrackbarPos("SAT max", "HSV")
    v_min = cv2.getTrackbarPos("VAL min", "HSV")
    v_max = cv2.getTrackbarPos("VAL max", "HSV")
    
    #print(h_min)
    
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])

    mask = cv2.inRange(imgHsv, lower, upper)
    #print(mask)    
    result = cv2.bitwise_and(img, img, mask = mask)
    
    #cv2.imshow("Original", img)
    #cv2.imshow("HSV Color Space", imgHsv)
    #cv2.imshow("HSV Mask", mask)
    #cv2.imshow("HSV result", result)
    from Stack_Image import stackImages
    StackImages = stackImages(([img, mask, result], [img, imgHsv, result]), 0.5)
    cv2.imshow("Stack Image",StackImages)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()









