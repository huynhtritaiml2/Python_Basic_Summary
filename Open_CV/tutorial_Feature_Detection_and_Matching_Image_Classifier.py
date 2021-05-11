#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 21:07:21 2021

@author: tai
"""

import cv2
import numpy as np

path1 = "Resources/KinectSportsSeason2.jpg"  # for Training
path2 = "Resources/Kinect4.jpg" # for Testing

img1 = cv2.imread(path1)
img2 = cv2.imread(path2)

#img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

'''
Now we will initialize our detector . There are many types of detectors available. 
Some are free and some require license for commercial use. The most common ones include ORB, SIFT and SURF. 
We will be using the ORB detector since it is free to use . To learn more about different detectors you can 
visit the opencv documentation.
https://docs.opencv.org/master/db/d27/tutorial_py_table_of_contents_feature2d.html

# To create our ORB (Oriented FAST and Rotated BRIEF) detector we can simply write
Feature Detector to find the special feature of object, corner of buidling, NOT edge or sky or wall of buiding
because they are the same at a lot of point

ORB,  : free
SIFT and SURF : Not free
so we use ORB

'''

orb = cv2.ORB_create()
# By default the ORB detector will retain 500 features. We could define a desired value as an argument as well.
#orb = cv2.ORB_create(nfeatures = 1000)
kp1, des1 = orb.detectAndCompute(img1, None) # kp == Keypont, des = Descriptor
kp2, des2 = orb.detectAndCompute(img2, None)

print(len(kp1)) # 500 # <KeyPoint 0x7fe4d22bacc0>, <KeyPoint 0x7fe4d22bacf0>] , store address :)) 
print(des1.shape) # (500, 32)
print(des1[0]) 
# [ 24  55  62  32 170 107 180  78 228  96 250 226 223 156 226  32  38 159 238 129 135 241 135 112 240 161  43 139 101  81 175   3]


# Kp1 are the Key Points of the query image and Kp2 are the Key points of the Training image. 
# To draw them we can use the opencv function drawKeypoints.
imgKp1 = cv2.drawKeypoints(img1, kp1, None)
imgKp2 = cv2.drawKeypoints(img2, kp2, None)


# Find the matching descriptor: Brute-Force Matcher
print("******************* Brute-Force Matcher")
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k = 3) # k = 2, use k mean,
'''
print(len(matches)) # 500
print(matches[0]) # [<DMatch 0x7fa54afee810>, <DMatch 0x7fa54afee830>]
print(matches[0][0].distance) # 391.1086730957031
print(matches[0][0].trainIdx) # 354
print(matches[0][0].queryIdx) # 0
print(matches[0][0].imgIdx) # 0

print(matches[0]) # [<DMatch 0x7fa54afee810>, <DMatch 0x7fa54afee830>]
print(matches[0][1].distance) # 392.13006591796875
print(matches[0][1].trainIdx) # 47
print(matches[0][1].queryIdx) # 0
print(matches[0][1].imgIdx) # 0

print(matches[1]) # [<DMatch 0x7fa54afee810>, <DMatch 0x7fa54afee830>]
print(matches[1][0].distance) # 390.2896423339844
print(matches[1][0].trainIdx) # 50
print(matches[1][0].queryIdx) # 1
print(matches[1][0].imgIdx) # 0

'
Nói chung là Trích từ ảnh test ra nFeature, rồi tính từng điêm feature đó với train_image,
-> Features in Test Image are in order
-> Features in Train Image are NOT in order
imgIdx: == 0 do chỉ có 1 train_image
'''

good = []
for m, n,_ in matches: # Because we have k = 2, 2 points near to it
    if m.distance < 0.75 * n.distance:
        good.append([m])

print("Number of match", len(good))

#imgBf = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches[0:1], None, flags = 2)
imgBf = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags = 2)


cv2.imshow("img training", imgKp1)
cv2.imshow("img testing", imgKp2)
cv2.imshow("img Brute-Force Matcher", imgBf)

cv2.waitKey(0)
cv2.destroyAllWindows()


'''
Reference:
    https://www.murtazahassan.com/courses/opencv-with-python-for-beginners/lesson/feature-detection-and-matching/
    
More kind of Distacne between 2 feature:
    https://medium.com/swlh/different-types-of-distances-used-in-machine-learning-ec7087616442
    
OpenCv Reference: 
    https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html
'''