#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 10:12:10 2021

@author: tai
"""

import cv2
import numpy as np
import os


path = "ImagesTrain"
orb = cv2.ORB_create(nfeatures = 1000)

##### Import Images #####
images = []
classNames = []
myList = os.listdir(path)
print("Total Classes Detected", len(myList))

for cl in myList:
    imgCur = cv2.imread(f"{path}/{cl}")
    imgCur = cv2.cvtColor(imgCur, cv2.COLOR_BGR2GRAY)
    images.append(imgCur)
    classNames.append(os.path.splitext(cl)[0]) # To take the name of file NOT extension
    
print(classNames)


'''
findDes:
    find the Descriptor for each image
'''

def findDes(images):
    desList = []
    for img in images:
        kp, des = orb.detectAndCompute(img, None)
        desList.append(des)
    return desList

desList = findDes(images)
print(len(desList)) # 3

'''
findID:
    find Hamming Distance (k smallest distance) for Test_image and 3 Train_image (3, nfeatures = 1000)
    then, find good_point from 2 pair of images
    then, choose the 2 pair of images have larger number of good_point.
'''
def findID(img, desList):
    kp2, des2 = orb.detectAndCompute(img, None) # Find the Feature from image
    # Find the matching descriptor: Brute-Force Matcher
    # print("******************* Brute-Force Matcher")
    bf = cv2.BFMatcher()
    matchList = [] # Store all Hamming Distance for each pair (Test_image and Train_image)
    finalVal = -1 # Store the index/ name of class. Take from name of Traing_image
    
    try: # Some times, ERROR Because, No matching between 2 images
        for des in desList:
            matches = bf.knnMatch(des, des2, k = 2) # k = 2, use k mean,
            good = []
            for m, n in matches: # Because we have k = 2, 2 points near to it
                if m.distance < 0.75 * n.distance:
                    good.append([m])
            #print(len(good))
            matchList.append(len(good))
    except:
         pass
    print(matchList)
     
    '''
    Problem: Đôi khi matchList = [16, 20, 7] thì nó quá ít để được coi là matching.
    '''
    threshold = 20 
    if len(matchList) != 0: 
        if max(matchList) > threshold:
            finalVal = matchList.index(max(matchList))

    return finalVal



frameWidth = 640
frameHeight = 480

# Test Video
path_video = "ImagesTest/game_cd_video.mp4"
path_video = "ImagesTest/Obama_Elon_musk_Video.mp4"
#cap = cv2.VideoCapture(path_video)

cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)

while True:
    success, img2 = cap.read()
    
    #'''
    # Test for Image
    path_test = "ImagesTest/Kinect.jpg"
    path_test = "ImagesTest/Det.jpg"
    path_test = "ImagesTest/Last.jpg"
    path_test = "ImagesTest/Obama_test.jpg"
    path_test = "ImagesTest/Elon_musk_test.jpg"
    img2 = cv2.imread(path_test)
    #'''
    
    imgOriginal = img2.copy()
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    
    id = findID(img2, desList)
    if id != -1:
        cv2.putText(imgOriginal, str(classNames[id]), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow("img2", imgOriginal)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



"""
Problem: image is 2d, NOT 3D object

"""





"""

path1 = "Resources/KinectSportsSeason2.jpg"  # for Training
path2 = "Resources/Kinect4.jpg" # for Testing

img1 = cv2.imread(path1)
img2 = cv2.imread(path2)

#img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None) # kp == Keypont, des = Descriptor
kp2, des2 = orb.detectAndCompute(img2, None)

print(len(kp1)) # 500 # <KeyPoint 0x7fe4d22bacc0>, <KeyPoint 0x7fe4d22bacf0>] , store address :)) 
print(des1.shape) # (500, 32)
print(des1[0]) 

imgKp1 = cv2.drawKeypoints(img1, kp1, None)
imgKp2 = cv2.drawKeypoints(img2, kp2, None)


# Find the matching descriptor: Brute-Force Matcher
print("******************* Brute-Force Matcher")
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k = 3) # k = 2, use k mean,


good = []
for m, n,_ in matches: # Because we have k = 2, 2 points near to it
    if m.distance < 0.75 * n.distance:
        good.append([m])

print("Number of match", len(good))

imgBf = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags = 2)



cv2.imshow("img training", imgKp1)
cv2.imshow("img testing", imgKp2)
cv2.imshow("img Brute-Force Matcher", imgBf)

cv2.waitKey(0)
cv2.destroyAllWindows()


"""
















