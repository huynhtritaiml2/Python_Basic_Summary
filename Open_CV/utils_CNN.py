#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 11:30:46 2021

@author: tai
"""

import pandas as pd
import numpy as np
import os

'''
getName: get the name of file from the Path
'''
def getName(filePath):
    # return filePath.split('\\')[-1] # For Windows
    return filePath.split('/')[-1] # For Ubuntu

def importDataInfo(path):
    columns = ["Center", "Left", "Right", "Steering", "Throttle", "Brake", "Speed"]
    data = pd.read_csv(os.path.join(path, "driving_log.csv"), names= columns)
    #print(data.head())
    #print(data["Center"][0])
    #print(getName(data["Center"][0]))
    '''
                                                      Center  ...     Speed
    0  /home/tai/Downloads/Python_basic_summary/Open_...  ...  25.94534
    1  /home/tai/Downloads/Python_basic_summary/Open_...  ...  25.68483
    2  /home/tai/Downloads/Python_basic_summary/Open_...  ...  25.47830
    3  /home/tai/Downloads/Python_basic_summary/Open_...  ...  25.22250
    4  /home/tai/Downloads/Python_basic_summary/Open_...  ...  25.01970
    
    [5 rows x 7 columns]
    /home/tai/Downloads/Python_basic_summary/Open_CV/Seft_Driving_Car_Data/IMG/center_2021_04_27_10_42_13_450.jpg
    center_2021_04_27_10_42_13_450.jpg
    '''
    
    data["Center"] = data["Center"].apply(getName) # ************ New way :))) data["Center"] = getName(data["Center"]
    #print(data.head())
    '''
                                   Center  ...     Speed
    0  center_2021_04_27_10_42_13_450.jpg  ...  25.94534
    1  center_2021_04_27_10_42_13_541.jpg  ...  25.68483
    2  center_2021_04_27_10_42_13_633.jpg  ...  25.47830
    3  center_2021_04_27_10_42_13_723.jpg  ...  25.22250
    4  center_2021_04_27_10_42_13_814.jpg  ...  25.01970
    '''
    
    print("Total Images Imported:", data.shape[0]) # Total Images Imported: 6704 images from Center_camera
    
    return data


################# STEP 2: Visualize , and Pre-Processing / Balancing the data
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
def balanceData(data, display = True):
    nBins = 31 # Choose odd number because we want zero at the center, and positive and negative sides
    samplesPerBin = 1000 # Maximum Samples per Bins
    hist, bins = np.histogram(data["Steering"], nBins) 
    # Note: bins in the limit of each bar :)) 
    #print(bins)
    #print(len(hist)) # 31
    #print(hist)
    '''
    [-1.         -0.93548387 -0.87096774 -0.80645161 -0.74193548 -0.67741935
     -0.61290323 -0.5483871  -0.48387097 -0.41935484 -0.35483871 -0.29032258
     -0.22580645 -0.16129032 -0.09677419 -0.03225806  0.03225806  0.09677419
      0.16129032  0.22580645  0.29032258  0.35483871  0.41935484  0.48387097
      0.5483871   0.61290323  0.67741935  0.74193548  0.80645161  0.87096774
      0.93548387  1.        ]
    NOTE: 32 values
    Problem: do not have bins for 0
    Solution:
    '''
    if display:
        center = (bins[: -1] + bins[1:])
        #print(center)    
        '''
        [-1.93548387 -1.80645161 -1.67741935 -1.5483871  -1.41935484 -1.29032258
         -1.16129032 -1.03225806 -0.90322581 -0.77419355 -0.64516129 -0.51612903
         -0.38709677 -0.25806452 -0.12903226  0.          0.12903226  0.25806452
          0.38709677  0.51612903  0.64516129  0.77419355  0.90322581  1.03225806
          1.16129032  1.29032258  1.41935484  1.5483871   1.67741935  1.80645161
          1.93548387]
        Problem: they are in nearly double
        Solution:
        '''
        center = (bins[: -1] + bins[1:]) * 0.5
        #print(center) 
        '''
        [-0.96774194 -0.90322581 -0.83870968 -0.77419355 -0.70967742 -0.64516129
         -0.58064516 -0.51612903 -0.4516129  -0.38709677 -0.32258065 -0.25806452
         -0.19354839 -0.12903226 -0.06451613  0.          0.06451613  0.12903226
          0.19354839  0.25806452  0.32258065  0.38709677  0.4516129   0.51612903
          0.58064516  0.64516129  0.70967742  0.77419355  0.83870968  0.90322581
          0.96774194]
        NOTE: Now, we have ZERO at the center of histogram
        '''
        
        plt.bar(center, hist, width = 0.06) # width is width of bar, not space between data number :)) ***
        plt.show()
        '''
        Problems: Too many value of Zeros, because the road is straight ==> Unbalance data in Sterring Angle
        '''
        # Find the cutoff point for balancing the data
        plt.bar(center, hist, width = 0.06)
        plt.plot((-1, 1), (samplesPerBin, samplesPerBin))
        plt.show()

    # Remove the redundant index
    '''
    Problems: they maybe in order, this code will delete all small value and keep all big value
    Solutions: shuffle before remove
    Thuật toán này là:
        1. giới hạn cho mỗi bin
        2. tìm index nếu steering trong giới hạn đó 
        3. shufle
        4. loại bỏ nếu nhiều hơn 500 :)) , nhỏ hơn thì giữ nguyên
    '''
    removeIndexList = []
    for j in range(nBins):
        binDataList = []
        for i in range(len(data["Steering"])):
            if data["Steering"][i] >= bins[j] and data["Steering"][i] <= bins[j+1]: 
                # NOTE: bins is limit of each bins
                # if Steering value in limit of a bin
                binDataList.append(i)
                
        binDataList = shuffle(binDataList)
        binDataList = binDataList[samplesPerBin:] # Remove above the line
        removeIndexList.extend(binDataList) # similar to append, but only take value, not type (ex: tupple, set)
        
    print("Removed Images: ", len(removeIndexList)) # Removed Images:  4125
    #print(removeIndexList)
    #print(data.index[removeIndexList])
    #data.drop(data.index[removeIndexList], inplace = True) # Method 1: FUCKING :))
    data.drop(removeIndexList, inplace = True) # Method 2: shorter 
    print("Remaining Images: ", len(data)) # Remaining Images:  2579

    if display:
        hist, _ = np.histogram(data["Steering"], nBins)
        plt.bar(center, hist, width = 0.06)
        plt.plot((-1, 1), (samplesPerBin, samplesPerBin))
        plt.show()
    '''
    NOTE: now, we have the good amount of data
    '''

    return data

############# STEP 3: LOAD DATA 
'''
1. Create correct path for each images
'''
def loadData(path, data):
    imagesPath = []
    steering = []
    for i in range(len(data)):
        indexedData = data.iloc[i] # take each entries from the data
        #print(indexedData)
        
        imagesPath.append(os.path.join(path, "IMG", indexedData[0]))
        #print(os.path.join(path, "IMG", indexedData[0]))
        '''
        Self_Driving_Car_Data/IMG/center_2021_04_27_11_02_06_290.jpg
        Self_Driving_Car_Data/IMG/center_2021_04_27_11_02_06_504.jpg
        ...
        
        
        '''
        
        steering.append(float(indexedData[3]))
        
    imagesPath = np.asarray(imagesPath)
    #print(type(imagesPath[0])) # <class 'numpy.str_'>
    steering = np.asarray(steering)
    
    return imagesPath, steering


########### STEP 4: Split into Training and Validation
########### STEP 5: Agumentation data
import matplotlib.image as mpimg # RGB images, NOT BGR in opencv
from imgaug import augmenters as iaa
import cv2
def augmentImage(imgPath, steering):
   
    img = mpimg.imread(imgPath)
    #print(np.random.rand(),np.random.rand(),np.random.rand(),np.random.rand())
    ## PAN: Shift right or left by 10%
    if np.random.rand() < 0.5:
        pan = iaa.Affine(translate_percent = {'x': (-0.1, 0.1), 'y': (-0.1, 0.1)})
        img = pan.augment_image(img)
    ### ZOOM
    if np.random.rand() < 0.5:
        zoom = iaa.Affine(scale = (1, 1.2)) # scale from 1 ->1.2
        img = zoom.augment_image(img) 
    ### BRIGHTNESS
    if np.random.rand() < 0.5:
        brightness = iaa.Multiply((0.2, 1.2)) # change birght from 0.2 to 1.2
        img = brightness.augment_image(img)
    
    ### FLIP
    '''
    NOTE: If we flip image, we need to change the sign of the steering value
    '''
    if np.random.rand() < 0.5:
        img = cv2.flip(img, 1)
        steering = -steering
    
    '''
    Problem: we do not want to add all augment, 
    Solution: random choose with all option == 0.5 probability
    '''
    
    
    
    return img, steering

'''    
imgRe, st = augmentImage("Self_Driving_Car_Data/IMG/center_2021_04_27_10_42_13_999.jpg", 0)
plt.imshow(imgRe)
plt.show()
plt.imshow(mpimg.imread("Self_Driving_Car_Data/IMG/center_2021_04_27_10_42_13_999.jpg"))
plt.show()
'''



########### STEP 6: Pro-Processing, creat Batch Generator
'''
Problem: we only want crop the road region, delete backgroud (tree, sky)
1. Crop
2. Change to YUV
3. Resize to NVIDIA model
4. Normalization
'''
def preProcessing(img):
    img = img[60:135, : , :] # Crop the road region
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3,3), 0)
    img = cv2.resize(img, (200, 66)) # Because NVIDIA use this size (220, 66)
    
    # Normalization: change value from 0->255 to 0 ->1
    img = img / 255
    
    
    return img
'''
img = preProcessing(mpimg.imread("Self_Driving_Car_Data/IMG/center_2021_04_27_10_42_13_999.jpg", 0))
plt.imshow(img)
plt.show()
plt.imshow(mpimg.imread("Self_Driving_Car_Data/IMG/center_2021_04_27_10_42_13_999.jpg"))
plt.show()
'''


'''
Ex: Take 100 image Paths, then augumented -> batchSize images, then pass throught the model 
NOTE: Batch_Size image NOT 100-> 300 or 400 :)) FUCING
NOTE 2: Batch_Size may have the same image, BUT with different augmented
NOTE: We have different Batch_Size during a Epoch and EVEN in Different EPOCH :)) 
    Batch_SIZE always different, random created
'''

import random
def batchGen(imagesPath, steeringList, batchSize, trainFlag):
    while True:
        imgBatch = []
        steeringBatch = []
        
        for i in range(batchSize):
            index = random.randint(0, len(imagesPath) - 1) # -1 : due to Index
            '''
            Because augment only for training_data
            '''
            if trainFlag:
                img, steering = augmentImage(imagesPath[index], steeringList[index])
            else:
                '''
                Because we use img = mpimg.imread(imgPath) in augmentImage, so if we do not use it
                we need to read image only
                '''
                img = mpimg.imread(imagesPath[index])
                steering = steeringList[index]
            
            img = preProcessing(img)
            imgBatch.append(img)
            steeringBatch.append(steering)
        yield(np.asarray(imgBatch), np.asarray(steeringBatch))
        '''
        We need convert into np.array similar to loadData() function
        '''
    
########### STEP 7: Create and compile the model of NVIDIA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
'''
Input : 66x200x3 # road region
1. Normalization

'''
def createModel():
    model = Sequential()
    
    # kernel = 5x5, nFilter = 24, stride = 2, input_shape, activation = elu ("elu) better than relu in this case
    # Stride = 2, to reduce the size of image
    model.add(Convolution2D(24, (5, 5), (2, 2), input_shape = (66, 200, 3), activation ="elu"))
    model.add(Convolution2D(36, (5, 5), (2, 2), activation ="elu"))
    model.add(Convolution2D(48, (5, 5), (2, 2), activation ="elu"))
    model.add(Convolution2D(64, (3, 3), (1, 1), activation ="elu"))
    model.add(Convolution2D(64, (3, 3), (1, 1), activation ="elu"))
    
    model.add(Flatten()) # 1164 neurons
    model.add(Dense(100, activation = "elu"))
    model.add(Dense(50, activation = "elu"))
    model.add(Dense(10, activation = "elu"))
    model.add(Dense(1))
    
    model.compile(Adam(lr = 0.0001), loss = "mse") # use MSE because it is Regression Problem
    
    return model

########### STEP 8: Training model
########### STEP 9: Save model


