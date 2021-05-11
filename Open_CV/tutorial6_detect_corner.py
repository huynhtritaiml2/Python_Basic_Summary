#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 13:11:26 2021

@author: tai
"""
"""
import numpy as np
import cv2

img = cv2.imread("assets/chessboard.png")
img = cv2.resize(img, (0,0), fx=0.75, fy=0.75)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # easier to detect the corner

N = 100 # N: number of the best corners
isCorners = 0.01 # 0->1 : depend on algorithm, Minimum Quality
minimum_distance = 10 # Minimum Euclidean Distance
corners = cv2.goodFeaturesToTrack(gray, N, isCorners, minimum_distance) # return Floating-point
corners = np.int0(corners) # Convert to integer int64, Hơi khó Hiểu 
#print(corners)

for corner in corners:
    x, y = corner.ravel() # [[[0, 1, 2]]] -> [0, 1, 2] , [[1, 2], [2, 1]] -> [1, 2, 2, 1]
    cv2.circle(img, (x,y), 5, (255, 0, 0), -1)
    

cv2.imshow("Frame",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

######################################## Draw Line Between 2 corner ##########################
import numpy as np
import cv2

img = cv2.imread("assets/chessboard.png")
img = cv2.resize(img, (0,0), fx=0.75, fy=0.75)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # easier to detect the corner

N = 100 # N: number of the best corners
isCorners = 0.01 # 0->1 : depend on algorithm, Minimum Quality
minimum_distance = 10 # Minimum Euclidean Distance
corners = cv2.goodFeaturesToTrack(gray, N, isCorners, minimum_distance) # return Floating-point
corners = np.int0(corners) # Convert to integer int64, Hơi khó Hiểu 
#print(corners)

for corner in corners:
    x, y = corner.ravel() # [[[0, 1, 2]]] -> [0, 1, 2] , [[1, 2], [2, 1]] -> [1, 2, 2, 1]
    cv2.circle(img, (x,y), 5, (255, 0, 0), -1)
    
for i in range(len(corners)):
    for j in range(i + 1 , len(corners)):
        corner1 = tuple(corners[i][0])
        corner2 = tuple(corners[j][0])
        color = tuple(map(lambda x: int(x), np.random.randint(0, 255, size = 3))) # Method 1
        color = tuple(int(x) for x in np.random.randint(0, 255, size = 3)) # Method 2
        #color = tuple(np.random.randint(0, 255, size = 3).astype(np.ubyte)) # Not Woking
        '''
        Có vẻ không có nhiều cách để tránh dùng int() của Python, data structure của nó khác.
        '''
        #print(color)
        cv2.line(img, corner1, corner2, color, 1)
        
cv2.imshow("Frame",img)
cv2.waitKey(0)
cv2.destroyAllWindows()