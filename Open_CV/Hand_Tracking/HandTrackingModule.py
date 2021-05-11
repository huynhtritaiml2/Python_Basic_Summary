#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 18:00:09 2021

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

class handDetector():
    def __init__(self, mode = False, maxHands = 2, detectionCon = 0.5, trackCon = 0.5):
        '''
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5):
        '''
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        
        # Config model
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, 
                                        self.detectionCon, self.trackCon) # Use default values, 
        self.mpDraw = mp.solutions.drawing_utils # Draw 21 point of hands, and connect them
        
    
    
    def findHand(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Because hands use RGB images
        self.results = self.hands.process(imgRGB)
        
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                #mpDraw.draw_landmarks(img, handLms) # Point in the hand, do not have any line conect
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img
    
    
    '''
    Find position of the first_hand, return list of position of 21 finger
    '''
    def findPosition(self, img, handNo = 0, draw = True):
        lmList = [] # landmarks list we will return
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                
                h, w, c = img.shape
                # Convert landmark into real pixel position
                cx, cy = int(lm.x * w), int(lm.y * h)
                # Draw a circle at landmark 0
                lmList.append([id, cx, cy])
                if draw: # Use the id to track any finger
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

        return lmList
    
################################### Test Module ###################################
def main():
    cap = cv2.VideoCapture(0)
    
    pTime = 0
    cTime = 0 # Current time
    detector = handDetector() # Create Object
    
    while True:
        succes, img = cap.read()
        
        img = detector.findHand(img)
        lmList = detector.findPosition(img)
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
        
if __name__ == "__main__":
    main()