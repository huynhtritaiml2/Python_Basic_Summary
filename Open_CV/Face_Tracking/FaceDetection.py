#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 03:51:11 2021

@author: tai
"""

'''
Run on CPU and modbile devices of google
Video have higher FPS than Webcam, then we use Video
'''

import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture("Videos/2.mp4")
pTime = 0

mpFaceDetection = mp.solutions.face_detection
faceDetection = mpFaceDetection.FaceDetection()

mpDraw = mp.solutions.drawing_utils


while True:
    success, img = cap.read()
    
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    #print(results) # <class 'mediapipe.python.solution_base.SolutionOutputs'> # FPS reduce when we use print
    
    # Check for availabel
    if results.detections:
        for id, detection in enumerate(results.detections):
            #print(id, detection)
            '''
            0  # Order of face
            label_id: 0
            score: 0.8438937067985535 # Score
            location_data { # BOUNDING BOX
              format: RELATIVE_BOUNDING_BOX
              relative_bounding_box {
                xmin: 0.46737608313560486
                ymin: 0.14302480220794678
                width: 0.13135668635368347
                height: 0.23352324962615967
              }
              relative_keypoints {
                x: 0.5010300874710083
                y: 0.20848822593688965
              }
              relative_keypoints {
                x: 0.5460773706436157
                y: 0.21581608057022095
              }
              relative_keypoints {
                x: 0.5148670673370361
                y: 0.27059406042099
              }
              relative_keypoints {
                x: 0.5166276693344116
                y: 0.316699743270874
              }
              relative_keypoints {
                x: 0.4843997359275818
                y: 0.21809417009353638
              }
              relative_keypoints {
                x: 0.5887718200683594
                y: 0.2352229356765747
              }
            }
            
            NOTE: we have 6 keypoints
            NOTE 2: RELATIVE_BOUNDING_BOX, mean ratio of 
            
            '''
            #print(detection.score)
            #print(detection.location_data.relative_bounding_box)
            '''
            [0.528350293636322]
            xmin: 0.4473288953304291
            ymin: 0.40650510787963867
            width: 0.1945943534374237
            height: 0.3459429144859314
            '''
            
            # Draw by using the function # Not use Default Function,
            #mpDraw.draw_detection(img, detection) # The rectangle and 6 points ????
            
            # Extract bouding box from Object
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            bbox =  int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                    int(bboxC.width * iw) , int(bboxC.height * ih) 
            
            cv2.rectangle(img, bbox, (255, 0, 255), 2)
            cv2.putText(img, f"{int(detection.score[0] * 100)}%", (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN, \
                        2, (255, 0, 255), 2)

    
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
    
    
    cv2.imshow("Video", img)
    if cv2.waitKey(1) & 0xFF == ord('q'): # Reduce FPS by cahnge waitKey(), 1: for RealTime
        break
    
cap.release()
cv2.destroyAllWindows()










