#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 17:04:10 2021

@author: tai
"""

import cv2

cap = cv2.VideoCapture(0)

# pip install opencv-contrib-python
tracker = cv2.legacy_TrackerMOSSE.create() # ???????????
tracker = cv2.legacy_TrackerCSRT.create() # Accuracy is high, but slower 
success, img = cap.read()
bbox = cv2.selectROI("Tracking", img, False) # ???????????
tracker.init(img, bbox) # ???????????


def drawBox(img, bbox):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(img, (x,y), ((x+w), (y+h)), (255, 0, 255), 3, 1)
    cv2.putText(img, "Tracking", (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


while True:
    timer = cv2.getTickCount()
    success, img = cap.read() 
    
    success, bbox = tracker.update(img) # ???????????
    print(type(bbox)) # <class 'tuple'>, need to convert to integer
    print(bbox) # (107.0, 246.0, 400.0, 180.0) == (x, y, w, h)

    if success:
        drawBox(img, bbox)
    else:
        cv2.putText(img, "Loss", (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        
    '''
    cv2.getTickFrequency() : Frame per Second
    cv2.getTickCount() : time
    '''
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    
    cv2.putText(img, str(int(fps)), (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("Tracking", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

cap.release()
cv2.destroyAllWindows()

'''
Recommend Resources:
    https://ehsangazar.com/object-tracking-with-opencv-fd18ccdd7369
    

'''