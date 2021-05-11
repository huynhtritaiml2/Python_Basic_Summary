#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 13:57:04 2021

@author: tai
"""

import cv2
import numpy as np

frameWidth, frameHeight = 640, 480

path = "Resources/file3.mp4"
cap = cv2.VideoCapture(path)

#cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)


def empty(a):
    pass

cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters", 640, 240)
cv2.createTrackbar("Threshold1", "Parameters", 0, 250, empty)
cv2.createTrackbar("Threshold2", "Parameters", 150, 250, empty)
cv2.createTrackbar("Min Area", "Parameters", 5000, 30000, empty)



def getContours(img, imgContour):
    '''
    RETR : Retrieval method, retrieve only the extreme outer corners
    cv2.RETR_TREE: retrieve all the contour and reconstruct of full hierarchy
    cv2.CHAIN_APPROX_NONE: get all points
    cv2.CHAIN_APPROX_SIMPLE:  it will compress the values and we will get lesser number of points
    Retrieval: Truy xuất, thu hồi, lấy ra 
    hierarchy: theo cấp bậc
    
    ************************** Khó hiểu, nên tách ra làm riêng lẻ ????????????????????????????
    '''
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # ??????????????????
    #cv2.drawContours(imgContour, contours, -1, (255, 0, 255), 7)


    # Remove smaller area, or noise
    for cnt in contours:
        area = cv2.contourArea(cnt)
        min_area = cv2.getTrackbarPos("Min Area", "Parameters")
        if area > min_area:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 7) # ??????????????????
            # Find corner point
            # Step 1: find the length
            # cv2.arcLength(cnt, Is_contour_closed = True)
            peri = cv2.arcLength(cnt, True) # ??????????????????
            # approx = cv2.approxPolyDP(cnt, resolution, Is_contour_closed)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True) # ??????????????????
            print(len(approx)) # 4, maybe rectangle or Square
            
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(imgContour, (x, y), (x+w, y+h), (0, 255, 0), 5)
            
            
            # Show more information
            cv2.putText(imgContour, "Points: " + str(len(approx)), (x + w +20, y + 20), 
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(imgContour, "Area: " + str(int(area)), (x + w +20, y + 45), 
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)

while True:
    success, img = cap.read()
    #'''
    if success == False:
        path = "Resources/file2.mp4"
        cap = cv2.VideoCapture(path)
    # Resize if this is a video
    img = cv2.resize(img, (frameWidth, frameHeight))
    #'''
    
    # Convert to Gray scale
    imgBlur = cv2.GaussianBlur(img, (7,7), 1)
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
    # Edge Dectection: we need a trackbar
    threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
    threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")
    imgCanny = cv2.Canny(imgGray, threshold1, threshold2)
    
    # Filter out all the noise and overlape, using Dilation
    kernel = np.ones((5, 5))
    imgDil = cv2.dilate(imgCanny, kernel, iterations =1)
    
    imgContour = img.copy()
    getContours(imgDil, imgContour)
    
    from Stack_Image import stackImages
    StackImages = stackImages(([img, imgGray, imgCanny], [imgContour, imgBlur, imgGray]), .2)
    cv2.imshow("Stack Images", StackImages)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()