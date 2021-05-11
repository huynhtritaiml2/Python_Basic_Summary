#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 10:10:23 2021

@author: tai
"""

import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    width = int(cap.get(3))
    height = int(cap.get(4))
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    #cv2.imshow("frame", frame)
    cv2.imshow("frame", hsv)
    
    if cv2.waitKey(1) == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()

######################### HSV Video ###############################
'''
HSV used to extract the color we want to extract
'''
import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    width = int(cap.get(3))
    height = int(cap.get(4))
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    #cv2.imshow("frame", frame)
    cv2.imshow("frame", hsv)
    
    if cv2.waitKey(1) == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()

######################### HSV Video ###############################
'''
HSV used to extract the color we want to extract
we need to define:
    upper bound:
    lower bound:
out of range is 0
in range is keep
1. Google: HSV color picker
2. We can find it by using:
        
'''
import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read() # (480, 640, 3)
    width = int(cap.get(3))
    height = int(cap.get(4))
    

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # (480, 640, 3)
    lower_blue = np.array([90, 50, 50]) # Original: [110, 50, 50]
    upper_blue = np.array([130, 255, 255])
    
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Note: bitwise_and take 2 arguiment
    result = cv2.bitwise_and(frame, frame, mask = mask)
    '''
    1 1 = 1
    0 1 = 0
    1 0 = 0
    0 0 = 0
    '''
    #cv2.imshow("frame", frame)
    cv2.imshow("frame", result)
    cv2.imshow("mask", mask)
    
    if cv2.waitKey(1) == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()

'''
import numpy as np
import cv2
#cv2.cvtColor([[[255, 0, 0]]], cv2.COLOR_BGR2HSV) # ERROR: TypeError: Expected Ptr<cv::UMat> for argument 'src'

BGR_color = np.array([[[255, 0, 0]]])
#cv2.imshow("BRG frame", BGR_color)  
x = cv2.cvtColor(BGR_color, cv2.COLOR_BGR2HSV)
    
cv2.destroyAllWindows()
'''