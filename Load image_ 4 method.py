#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 13:43:14 2021

@author: tai
"""

# Reference: # https://www.pluralsight.com/guides/importing-image-data-into-numpy-arrays

# Load the Image
'''
Method 1: Pillow PIL:

input: PNG or JPEG
'''
################################ Load image ###############################

# load and show an image with Pillow
from PIL import Image
# Open the image form working directory
image = Image.open("3096_color.jpeg")
# Some details about the image
print(image.format) # JPEG
print(image.size) # (481, 321)
print(image.mode) # RGB
# show the image
image.show()

################################ Convert image into np.array ###############################
import numpy as np
X_train = np.asarray(image) # Method 1:
X_train = np.array(image) # Method 2:
print(type(X_train)) # <class 'numpy.ndarray'>
print(X_train.shape) # (321, 481, 3)

################################ Reverse np.array into  PIL ###############################
# create Pillow image from np.array
image_PIL = Image.fromarray(X_train)
print(type(image_PIL)) # <class 'PIL.Image.Image'>
print(image_PIL.mode) # RGB
print(image_PIL.size) # (481, 321)


################################ Save image ###############################
image_PIL.save('3096_color_PIL.jpeg')

'''
############################### Method 2: Matplotlib: ###############################
'''
################################ Load image ###############################
# load and display an image with Matplotlib
from matplotlib import image
from matplotlib import pyplot

# load image as pixel array
image = image.imread('3096_color.jpeg')
# summarize shape of the pixel array
print(image.dtype) # uint8 # 8-bit unsigned integers
print(image.shape) # (321, 481, 3)
# display the array of pixels as an image
pyplot.imshow(image)
pyplot.show()

################################ Convert image into np.array ###############################
import numpy as np
X_train = np.asarray(image) # Method 1:
X_train = np.array(image) # Method 2:
print(type(X_train)) # <class 'numpy.ndarray'>
print(X_train.shape) # (321, 481, 3)

################################ Save image ###############################
# Matplot lib ????????????


'''
############################### Method 3: Keras: ###############################
'''
print("##### Keras #####")
################################ Load image ###############################
from keras.preprocessing.image import load_img

# load the image in PIL format
image = load_img("3096_color.jpeg")
# report details about the image
print(type(image)) # <class 'PIL.JpegImagePlugin.JpegImageFile'>
print(image.format) # JPEG
print(image.mode) # RGB
print(image.size) # (481, 321)

image.show()

################################ Convert image PIL into np.array ###############################
from keras.preprocessing.image import img_to_array
image_array = img_to_array(image)
print(type(image_array)) # <class 'numpy.ndarray'>
print(image_array.dtype) # float32
print(image_array.shape) # (321, 481, 3)

################################ Reverse np.array into  PIL ###############################
from keras.preprocessing.image import array_to_img
image_PIL = array_to_img(image_array)
print(type(image_PIL)) # <class 'PIL.Image.Image'>

################################ Save image ###############################
from keras.preprocessing.image import save_img
save_img("3096_color_Keras.jpeg", image_array)


'''
############################### Method 4: Opencv: ###############################
'''
print("##### Keras #####")
################################ Load image ###############################
import cv2
image = cv2.imread("3096_color.jpeg")
image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # BGR -> RGB
print(type(image)) # <class 'numpy.ndarray'>
print(type(image_RGB)) # <class 'numpy.ndarray'>
################################ Convert image PIL into np.array ###############################
################################ Reverse np.array ###############################
################################ Save image ###############################
cv2.imwrite("3096_color_Opencv.jpeg", image_RGB)