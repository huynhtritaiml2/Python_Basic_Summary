#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 17:46:32 2021

@author: tai
"""

"""
################## Read Image
import cv2

img = cv2.imread("Resources/lena.png")

cv2.imshow("Lena", img)

cv2.waitKey(0) # mili second
"""


################## Read Video
"""
import cv2

frameWidth = 640
frameHeight = 360

cap = cv2.VideoCapture("Resources/testVideo.mp4")
#cap.set(3, frameWidth) # Not work for loading a video
#cap.set(4, frameHeight)

while True:
    success, img = cap.read() # Read a frame store into img
    
    img = cv2.resize(img, (frameWidth, frameHeight))
    cv2.imshow("Video",img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    
cv2.destroyAllWindows()
"""
####################### Read Webcam
import cv2

frameWidth = 640
frameHeight = 360

cap = cv2.VideoCapture(0) # 0: Webcam 0, 1: Webcam 1
cap.set(3, frameWidth)
cap.set(4, frameHeight)

while True:
    success, img = cap.read()
    cv2.imshow("Video", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


