#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 18:43:33 2021

@author: tai
"""

'''
YOLOv3:
    Classify and Localization
'''

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
    # The rstrip() method removes any trailing characters (characters at the end a string), space is the default trailing character to remove.
    # https://www.w3schools.com/python/ref_string_rstrip.asp
    #classNames = f.read().rstrip('\n').split('\n')
    print(classNames)
    #print(len(classNames))

'''
https://pjreddie.com/darknet/yolo/
YOLOv3 
YOLOv3-320	: 320x320 size of image
mAP: 51.5	
FLOPS: 38.97 Bn	
FPS: 45
'''
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

'''
findObject: 
1. find 
'''
def findObjects(outputs, img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []
    
    for output in outputs: # we have 3 outputs
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
    # cv2.dnn.NMSBoxes(bbox, confidence_prediction, Confiden_Threshold, nmsThreshold)
    indices = cv2.dnn.NMSBoxes(bbox, confs, confiden_threshold, nmsThreshold) # ??????????????
    print(indices) # [[0]]
    for i in indices:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        
        cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 255), 2)
        cv2.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%'
                    , (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        '''
        i: ong object in image, there are more than one object
        classIds: store Id of all object in images.
        classNames: find the name for that index
        '''


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
    #print(layerNames) # all network, too longggg
    '''
    ['conv_0', 'bn_0', 'leaky_1', 'conv_1' ... , 'conv_105', 'permute_106', 'yolo_106']
    '''
    
    layerNames2 = net.getUnconnectedOutLayers()
    #print(layerNames2) # Show index :)) FUNCKING
    '''
    [[200]
     [227]
     [254]]
    '''
    '''
    i[0] because i == [200], and i[0] = 200
    i[0] - 1 because index from 0->final
    '''
    outputNames = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    #print(outputNames) # ['yolo_82', 'yolo_94', 'yolo_106'] :YEAHHHHHHH :))))
    
    
    # Take Output from any layername :)) YEAHH
    outputs = net.forward(outputNames) # ??????????????????
    #print(len(outputs)) # 3
    #print(len(outputs[0])) # 300
    #print(type(outputs)) # <class 'list'>
    #print(type(outputs[0])) # <class 'numpy.ndarray'>
    #print(outputs[0].shape) # (300, 85)
    #print(outputs[1].shape) # (1200, 85)
    #print(outputs[2].shape) # (4800, 85) 
    '''
    output1 : (300, 85)
    output2 : (1200, 85)
    output3 : (4800, 85)
    
    Why 85?
    we have 80 classes, 5 for what ?
    x_center
    y_center
    width
    height
    the confidence: that there is an object present in the bouding box
    
    80: Probability of each classes
    '''
    #print(outputs[0][0])
    #print(outputs[0][1])
    #print(outputs[0][2])
    #print(outputs[0][299])
    '''
    [5.9080750e-02 3.0526480e-02 3.0766341e-01 2.8413904e-01 1.1190263e-08
     0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00
     0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00
     0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00
     0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00
     0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00
     0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00
     0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00
     0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00
     0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00
     0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00
     0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00
     0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00
     0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00
     0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00
     0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00
     0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00]
    
    x_center: 5.9080750e-02
    y_center: 3.0526480e-02 
    width: 3.0766341e-01
    height: 2.8413904e-01 
    confidence: 1.1190263e-08 :)) TOO Small
    
    Do not have any object :))
    use function ->
    '''
    findObjects(outputs, img)

    cv2.imshow("Video", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

























