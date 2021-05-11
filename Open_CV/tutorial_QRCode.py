#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 13:34:44 2021

@author: tai
"""

# authorize: Cho Phép
# Un-authorize: không cho phép

import cv2
import numpy as np
from pyzbar.pyzbar import decode


path = "QR_code/QRcode2.png"
img = cv2.imread(path)
img = cv2.resize(img, (640, 360))
code = decode(img)
print(code)
'''
[Decoded(data=b'Hello :)', type='QRCODE', rect=Rect(left=180, top=180, width=1239, height=1239), 
polygon=[Point(x=180, y=180), Point(x=180, y=1419), Point(x=1419, y=1419), Point(x=1419, y=180)])]
'''

############## Read Image QR CODE ##############
#'''
for barcode in decode(img):
    print(barcode.data) # b'Hello :)'
    print(barcode.type) # QRCODE
    print(barcode.rect) # Rect(left=180, top=180, width=1239, height=1239)
    print(barcode.polygon) # [Point(x=180, y=180), Point(x=180, y=1419), Point(x=1419, y=1419), Point(x=1419, y=180)]
    
    myData = barcode.data.decode("utf-8")
    print(myData) # Hello :)
    
    points = np.array([barcode.polygon]) # Convert List of Class point into np.array
    
    points = points.reshape((-1, 1, 2))
    
    cv2.polylines(img, [points], True, (255, 0, 255), 5)
    point_rect = barcode.rect
    cv2.putText(img, myData, (point_rect[0], point_rect[1]), cv2.FONT_HERSHEY_SIMPLEX , 0.9, (255, 0, 255), 2)
    cv2.imshow("Img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#'''



############## Read Video QR CODE ##############

frameWidth = 640
frameHeight = 360
path = "QR_code/QRCode_BarCode_Video.mp4"
cap = cv2.VideoCapture(path)
#cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)

while True:
    success, img = cap.read()
    
    for barcode in decode(img):
        #print(barcode.data)       
        myData = barcode.data.decode("utf-8")
        print(myData) 
        
        '''
        Use polygon: Because it matching all QR Code
        1. convert polygon into array: Chỉ là thủ tục 
        '''
        
        points = np.array([barcode.polygon]) # Convert List of Class point into np.array
        #print(barcode.polygon)
        #print(points)
        '''
        [Point(x=3, y=801), Point(x=127, y=806), Point(x=246, y=800), Point(x=249, y=718), Point(x=249, y=716), Point(x=123, y=715), Point(x=6, y=717)]
        [[[  3 801]
          [127 806]
          [246 800]
          [249 718]
          [249 716]
          [123 715]
          [  6 717]]]
        '''
        
        points = points.reshape((-1, 1, 2))
        #print(points)
        '''
        [[[  3 801]]
         [[127 806]]
         [[246 800]]
         [[249 718]]
         [[249 716]]
         [[123 715]]
         [[  6 717]]]
        '''
        #  cv2.polylines(img, array_point, IsClosed, color, thickness)
        cv2.polylines(img, [points], True, (255, 0, 255), 5)
        point_rect = barcode.rect
        cv2.putText(img, myData, (point_rect[0], point_rect[1]), cv2.FONT_HERSHEY_SIMPLEX , 0.9, (255, 0, 255), 2)
    
    
    cv2.imshow("Video", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
    















