#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 03:41:19 2021

@author: tai
"""

import cv2
import numpy as np

##################### Choose Threshold #####################
frameWidth = 640
frameHeight = 360
confiden_threshold = 0.5
nmsThreshold = 0.3 # ??????? the lower it is, the more aggressive it will be, aggressive: xâm lược, tấn công

##################### Choose Input Video/ Webcam #####################

path = "Resources/car_video3.mp4"
path = "Resources/file3.mp4"
path = "Resources/testVideo.mp4"

cap = cv2.VideoCapture(path)

#cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)


##################### Take all Names of Classes YOLO can Localization #####################
classesFile = "cocoNames/coco.names"
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().splitlines()
    classNames = f.read().rstrip('\n').split('\n')


##################### Configure the YOLO Network #####################
width_height = 320

modelConfiguration = "yolov3-320.cfg"
modelWeights = "yolov3-320.weights"

#modelConfiguration = "yolov3-tiny.cfg"
#modelWeights = "yolov3-tiny.weights"

# Create our network
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights) ### How about another network, not Darknet
# CPU:
#net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV) # ?????????????
#net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU) # CPU 

# GPU, but not work well :)) 
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)



def findObjects(outputs, img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []
    
    for output in outputs: # we have 3 outputs, output1 : (300, 85) output2 : (1200, 85) output3 : (4800, 85)
        for det in output: # [x_center, y_center, w, h, confident, 80_class_probability]
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            
            if confidence > confiden_threshold:
                '''
                convert to real scale of image, int() NOT floating point value
                wT, hT is the shape of real image == 320
                '''
                w, h = int(det[2] * wT), int(det[3] * hT)
                x, y = int(det[0] * wT - w/2), int(det[1] * hT - h/2)
                
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence)) # confidence should be floating point
                
    print(len(bbox))
    
    # Choose the best Bouding Box
    indices = cv2.dnn.NMSBoxes(bbox, confs, confiden_threshold, nmsThreshold) # ??????????????
    print(indices) # [[0]]
    for i in indices:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        
        cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 255), 2)
        cv2.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%'
                    , (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)



while True:
    success, img = cap.read()
    # Convert image to blob: đốm
    '''
    1/255 : devide image by 255
    set width and size of image = 320 for yolov3-320
    [0, 0, 0], 1, crop=False : set default value ?????????????????????????????
    '''
    blob = cv2.dnn.blobFromImage(img, 1/255, (width_height, width_height), [0, 0, 0], 1, crop=False)
    net.setInput(blob) # ????????/
    
    # Extract the output layer
    layerNames = net.getLayerNames()
    layerNames2 = net.getUnconnectedOutLayers()
    outputNames = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]    

    # Take Output from any layername :)) YEAHH
    outputs = net.forward(outputNames) # ??????????????????
    findObjects(outputs, img)

    cv2.imshow("Video", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

























