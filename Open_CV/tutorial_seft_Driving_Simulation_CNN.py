#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 08:16:15 2021

@author: tai
"""
'''

The car sim was built using an older version of Unity3D and we need to install that exact version.
Download Unity3D:
- https://beta.unity3d.com/download/f5287bef00ff/public_download.html
from the reference: https://kaigo.medium.com/how-to-install-udacitys-self-driving-car-simulator-on-ubuntu-20-04-14331806d6dd
1. sudo apt install gconf-service lib32gcc1 lib32stdc++6 libc6-i386 libgconf-2-4 npm
At OpenCV folder
2. sudo dpkg -i ./unity-editor_amd64-5.5.1xf1Linux.deb
- If you get error about unmet dependencies you may need to run, and retry
sudo apt --fix-broken install


Donwload Simulation:
1. https://github.com/udacity/self-driving-car-sim
Install Simulation:
Change mode for file to execute
1. sudo chmod +x ./beta_simulator_linux/beta_simulator.x86_64
Run the file
2. ./beta_simulator_linux/beta_simulator.x86_64

Traing Mode:
for collect the data
Autonomous Mode:
for run automatically

Path of (Left-Center-Right) Camera: 
steering angle : góc lái
throttle: Chân ga
brake: thắng 
speed: tốc độ

Guide for installing: 
    https://kaigo.medium.com/how-to-install-udacitys-self-driving-car-simulator-on-ubuntu-20-04-14331806d6dd
'''

'''
NOTE: for data, we ONLY use Center_Camera, Not Left and Right Cameras
'''


############# STEP 1 
from utils_CNN import *
path = "Self_Driving_Car_Data"
data = importDataInfo(path)


############ STEP 2: visualize the data
data = balanceData(data, display = True)

############# STEP 3: LOAD DATA 
imagesPath, steering = loadData(path, data)
print(imagesPath[0], steering[0])
'''
Self_Driving_Car_Data/IMG/center_2021_04_27_10_42_13_999.jpg -0.25
'''


########### STEP 4: Split into Training and Validation
from sklearn.model_selection import train_test_split

xTrain, xVal, yTrain, yVal = train_test_split(imagesPath, steering, test_size = 0.2, random_state = 5)
print("Tortal Training Images: ", len(xTrain)) # Tortal Training Images:  2063
print("Total Validation Images: ", len(xVal)) # Total Validation Images:  516

########### STEP 5: Agumentation data
'''
lighting, zoom, shft left, right, crop
10,000 -> 30,000 or 40,000
'''

########### STEP 6: Pro-Processing, creat Batch Generator
'''
Augmented during training process, not before it
'''

########### STEP 7: Create and compile the model of NVIDIA
'''
print("Setting UP")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # Set this to NOT show warning
'''
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true' # Magically work well :)) NO ERROR

model = createModel()
model.summary()
'''
Model: "sequential_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_10 (Conv2D)           (None, 31, 98, 24)        1824      
_________________________________________________________________
conv2d_11 (Conv2D)           (None, 14, 47, 36)        21636     
_________________________________________________________________
conv2d_12 (Conv2D)           (None, 5, 22, 48)         43248     
_________________________________________________________________
conv2d_13 (Conv2D)           (None, 3, 20, 64)         27712     
_________________________________________________________________
conv2d_14 (Conv2D)           (None, 1, 18, 64)         36928     
_________________________________________________________________
flatten_2 (Flatten)          (None, 1152)              0         
_________________________________________________________________
dense_6 (Dense)              (None, 100)               115300    
_________________________________________________________________
dense_7 (Dense)              (None, 50)                5050      
_________________________________________________________________
dense_8 (Dense)              (None, 10)                510       
_________________________________________________________________
dense_9 (Dense)              (None, 1)                 11        
=================================================================
Total params: 252,219
Trainable params: 252,219
Non-trainable params: 0
_________________________________________________________________
'''


########### STEP 8: Training model
'''
batchSize = 100
steps_per_epoch = 300

==> we train 30,000 images for epochs

batchSize = 100
validation_steps = 200

==> we validate 20,000 images
'''
history = model.fit(batchGen(xTrain, yTrain, batchSize = 100, trainFlag = True), steps_per_epoch = 300, epochs = 10,
          validation_data = batchGen(xVal, yVal, batchSize = 100, trainFlag = False), validation_steps = 200)

'''
model.fit(batchGen(xTrain, yTrain, 10, 1), steps_per_epoch = 20, epochs = 2,
          validation_data = batchGen(xVal, yVal, 10, 0), validation_steps = 20)
'''
########### STEP 9: Save model
model.save("model.h5") # Save the weights and architecture of the model
print("Model saved")

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.legend(["Training", "Validation"])
#plt.ylim([0, 1])
plt.title("Loss")
plt.xlabel("Epoch")
plt.show()


########### STEP 10: Test the model
'''
The code is specified to the simulator, so we will use thier code

'''