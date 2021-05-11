#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 15:11:17 2021

@author: tai
"""

'''
pip install mediapipe
https://github.com/google/mediapipe

See more application in it, More code in C++
'''
'''
Hand Tracking = Palm Detection + Hand Landmarks
Palm: lòng bàn tay
Landmarks: Cột mốc
    
/home/tai/anaconda3/envs/tf2/lib/python3.8/site-packages    
'''
import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)


# Config model
mpHands = mp.solutions.hands
hands = mpHands.Hands() # Use default values, 
'''
  def __init__(self,
               static_image_mode=False,
               max_num_hands=2,
               min_detection_confidence=0.5,
               min_tracking_confidence=0.5):
      
static_image_mode=False
-> Sometimes Track, Sometimes Detect depend on confidence level
if confidence < min_tracking_confidence, it will do Detection again

static_image_mode=True : Detection Mode
-> Slower
'''
mpDraw = mp.solutions.drawing_utils # Draw 31 point of hands, and connect them

pTime = 0
cTime = 0 # Current time

while True:
    succes, img = cap.read()
    
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Because hands use RGB images
    results = hands.process(imgRGB)
    
    '''
    Maybe more than 1 hand in the image
    '''
    #print(results) # <class 'mediapipe.python.solution_base.SolutionOutputs'>
    #print(results.multi_hand_landmarks)
    '''
    landmark {
      x: 0.47991782426834106
      y: 0.6885459423065186
      z: 0.12430720776319504
    }
    '''
    #print(len(results.multi_hand_landmarks)) # 1
    #print(type(results.multi_hand_landmarks)) # <class 'list'>
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            #mpDraw.draw_landmarks(img, handLms) # Point in the hand, do not have any line conect
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            
            for id, lm in enumerate(handLms.landmark):
                #print(id, lm)
                '''
                0 
                x: 0.7972462773323059
                y: 0.8384082317352295
                z: -4.3646450649248436e-05
                
                1 
                x: 0.7180896401405334
                y: 0.8062342405319214
                z: -0.042422160506248474
                
                20 
                x: 0.8849480152130127
                y: 0.3829914331436157
                z: -0.08664213865995407
                
                NOTE: they are ratio of the image, not real pixel value
                '''
                h, w, c = img.shape
                #print(img.shape) # (480, 640, 3)
                # Convert landmark into real pixel position
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy) 
                '''
                0 353 441
                1 360 406
                
                ...
                
                20 369 393
                '''
                
                # Draw a circle at landmark 0
                if id == 4: # Use the id to track any finger
                    cv2.circle(img, (cx, cy), 25, (255, 0, 255), cv2.FILLED)
        
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)
    
    
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
