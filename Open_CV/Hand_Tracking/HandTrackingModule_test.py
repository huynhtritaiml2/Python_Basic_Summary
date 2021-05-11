#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 20:27:30 2021

@author: tai
"""

import  HandTrackingModule as htm
import cv2
import time


cap = cv2.VideoCapture(0)

pTime = 0
cTime = 0 # Current time
detector = htm.handDetector() # Create Object

while True:
    succes, img = cap.read()
    
    #img = detector.findHand(img)
    #lmList = detector.findPosition(img)
    
    img = detector.findHand(img, draw = True)
    lmList = detector.findPosition(img, draw = False) # Not draw too large point******
    if len(lmList) != 0: # ERROR: index is out of range
        print(lmList[4])
    
    
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()  
