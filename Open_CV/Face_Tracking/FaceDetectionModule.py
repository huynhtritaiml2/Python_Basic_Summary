#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 04:41:45 2021

@author: tai
"""

import cv2
import mediapipe as mp
import time

class FaceDetector():
    def __init__(self, minDetectionCon = 0.5):
        self.minDetectionCon = minDetectionCon
        
        self.mpFaceDetection = mp.solutions.face_detection
        self.faceDetection = self.mpFaceDetection.FaceDetection(minDetectionCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findFaces(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        
        bboxs = []
        # Check for availabl
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
    
                # Extract bouding box from Object
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox =  int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                        int(bboxC.width * iw) , int(bboxC.height * ih) 
                
                bboxs.append([bbox, detection.score])
                #cv2.rectangle(img, bbox, (255, 0, 255), 2)
                if draw:
                    img = self.fancyDraw(img, bbox)
                    cv2.putText(img, f"{int(detection.score[0] * 100)}%", (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN, \
                            2, (255, 0, 255), 2)
    
        
        
        return img, bboxs
    '''
    Draw Target Rectable
    l : line_long of target
    t : line thickness 
    rt : rectangle thickness 
    '''
    def fancyDraw(self, img, bbox, l = 30, t = 5, rt = 1): # Each bbox at the time, NOT list
        
        x, y, w, h = bbox
        x1, y1 = x + w, y + h# Bottom Right Point of Rectangle 
        # NOTE: we use OpenCV to draw, NOT
        cv2.rectangle(img, bbox, (255, 0, 255), rt)
        
        # Top Left x,y
        cv2.line(img, (x, y), (x + l, y), (255, 0, 255), t)
        cv2.line(img, (x, y), (x , y + l), (255, 0, 255), t)
        
        # Top Right x1,y
        cv2.line(img, (x1, y), (x1 - l, y), (255, 0, 255), t) # We use -, minus
        cv2.line(img, (x1, y), (x1 , y + l), (255, 0, 255), t)
        
        # Bottom Left x,y1
        cv2.line(img, (x, y1), (x + l, y1), (255, 0, 255), t)
        cv2.line(img, (x, y1), (x , y1 - l), (255, 0, 255), t) # We use -, minus
        
        # Top Left x1,y1
        cv2.line(img, (x1, y1), (x1 - l, y1), (255, 0, 255), t) # We use -, minus
        cv2.line(img, (x1, y1), (x1 , y1 - l), (255, 0, 255), t) # We use -, minus
        
        return img
    


def main():
    cap = cv2.VideoCapture("Videos/2.mp4")
    pTime = 0
    
    mpFaceDetection = mp.solutions.face_detection
    faceDetection = mpFaceDetection.FaceDetection()
    mpDraw = mp.solutions.drawing_utils
    
    detector = FaceDetector(minDetectionCon = 0.6) # Default value = 0.5
    while True:
        success, img = cap.read()
        
        
        img, bbox = detector.findFaces(img, draw=False)
        img, bbox = detector.findFaces(img)
        #print(bbox) # [[(693, 183, 295, 295), [0.9456474781036377]]]
        
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime
        cv2.putText(img, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
        
        cv2.imshow("Video", img)
        if cv2.waitKey(1) & 0xFF == ord('q'): # Reduce FPS by cahnge waitKey(), 1: for RealTime
            break
        
    cap.release()
    cv2.destroyAllWindows()    

if __name__ == "__main__":
    main()
    