#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 09:07:22 2021

@author: tai
"""

import cv2
import numpy as np


def mousePoints(event, x, y, flags, params):
    global counter
    if event == cv2.EVENT_LBUTTONDOWN:
        #print(x, y)
        
        circles[counter] = x, y
        counter += 1
        print(circles, counter)

path = "Resources/book.jpeg"
img = cv2.imread(path)
#cv2.imshow("Original Image", img)


circles = np.zeros((4, 2), np.uint) # 4 vertex of the book cover
counter = 0 # index of the vertex
#cv2.setMouseCallback("Original Image", mousePoints)

while True:
    if counter == 4:
        width, height = 250, 350
        #point1 = np.float32([[29, 137], [260, 65], [229, 472], [518, 336]])
        point1 = np.float32(circles)
        point2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
        
        matrix = cv2.getPerspectiveTransform(point1, point2)
        imgOutput = cv2.warpPerspective(img, matrix, (width, height))
        
        cv2.imshow("Warp Perspective", imgOutput)
        
        for i in range(0, 4):
            cv2.circle(img, (point1[i, 0], point1[i, 1]), 5, (0, 0, 255), cv2.FILLED)
    
    cv2.imshow("Original Image", img)
    cv2.setMouseCallback("Original Image", mousePoints)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cv2.waitKey(0)
cv2.destroyAllWindows()

'''
NOTE: we can add more vertex or, RIGHT_BUTTON_CLICK for break :))
    
'''