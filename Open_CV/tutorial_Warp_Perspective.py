#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 08:36:13 2021

@author: tai
"""

import cv2
import numpy as np

# Warp Perspective == Bird views: Phối cảnh dọc, Làm cong bối cảnh 

path = "Resources/card.jpeg"
img = cv2.imread(path)

# Cordinate of 4 vertex of a K card in the image (x, y)
point1 = np.float32([[50, 100], [132, 84], [70, 222], [163, 203]])
print(point1)
'''
[[ 50. 100.]
 [132.  84.]
 [ 70. 222.]
 [163. 203.]]
'''
width, height = 250, 350
point2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
matrix = cv2.getPerspectiveTransform(point1, point2) # Matrix for transformation
imgOutput = cv2.warpPerspective(img, matrix, (width, height))


for i in range(0, 4):
    cv2.circle(img, (point1[i, 0], point1[i, 1]), 5, (0, 0, 255), cv2.FILLED) # Plot vertex 


cv2.imshow("Original Image",img)
cv2.imshow("Warp Perspective", imgOutput)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
How to get rectangle automatically without human intervention
intervention: sự can thiệp
'''