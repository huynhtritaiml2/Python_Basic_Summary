#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 14:01:18 2021

@author: tai
"""

import cv2
import numpy as np
from pyzbar.pyzbar import decode

frameWidth = 640
frameHeight = 360
path = "QR_code/QRCode_BarCode_Video.mp4"
cap = cv2.VideoCapture(path)
#cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)

with open("QRCode_Authorization.text") as f:
    myDataList = f.read().splitlines()
print(myDataList)
'''
['Hello :)', '0705632085943', 'https://suckhoetoandan.vn/khaiyte', 
 'https://baohodoanhnghiep.com', '0051111407592']
'''
while True:
    success, img = cap.read()
    
    for barcode in decode(img):
        myData = barcode.data.decode("utf-8")
        
        # Check Authorization
        if myData in myDataList:
            #print("Authorized")
            myOutput = "Authorized"
            myColor = (0, 255, 0)
            
        else:
            #print("Un-Authorized")
            myOutput = "Un-Authorized"
            myColor = (0, 0, 255)
        
        points = np.array([barcode.polygon]) # Convert List of Class point into np.array
        points = points.reshape((-1, 1, 2))

        '''
        cv2.polylines(img, [points], True, (255, 0, 255), 5)
        point_rect = barcode.rect
        cv2.putText(img, myData, (point_rect[0], point_rect[1]), cv2.FONT_HERSHEY_SIMPLEX , 0.9, (255, 0, 255), 2)
        # Put Text Authorized or Un-Authorized
        cv2.putText(img, myOutput, (point_rect[0], point_rect[1]-20), cv2.FONT_HERSHEY_SIMPLEX , 0.9, (255, 0, 255), 2)
        '''
        
        # Add Text and color for Authorized and Un-Authorized
        cv2.polylines(img, [points], True, myColor, 5)
        point_rect = barcode.rect
        cv2.putText(img, myData, (point_rect[0], point_rect[1]), cv2.FONT_HERSHEY_SIMPLEX , 0.9, myColor, 2)
        # Put Text Authorized or Un-Authorized
        cv2.putText(img, myOutput, (point_rect[0], point_rect[1]-20), cv2.FONT_HERSHEY_SIMPLEX , 0.9, myColor, 2)

    
    cv2.imshow("Video", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()