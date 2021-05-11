#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 21:20:12 2021

@author: tai
"""

print('Setting UP')
'''
pip install python-socketio==4.2.1 # python-socketio 4.2.1
pip install eventlet
pip install Flask

Run
1. Run this code
2. Run simulator: 
    /home/tai/Downloads/Python_basic_summary/Open_CV/term1-simulator-linux/beta_simulator_linux/beta_simulator.x86_64

'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import socketio # Install
import eventlet # Install
import numpy as np
from flask import Flask # Install
from tensorflow.keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import cv2
 
#### FOR REAL TIME COMMUNICATION BETWEEN CLIENT AND SERVER
sio = socketio.Server()
#### FLASK IS A MICRO WEB FRAMEWORK WRITTEN IN PYTHON
app = Flask(__name__)  # '__main__'
 
maxSpeed = 10
 
 
def preProcess(img): # Exactly from the previous lecture/code
    img = img[60:135,:,:]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,  (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img/255
    return img
 
 
@sio.on('telemetry')
def telemetry(sid, data):
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image']))) # Image from the car
    image = np.asarray(image)
    image = preProcess(image) # Preprocess before the testing
    image = np.array([image])
    steering = float(model.predict(image)) # Predict Steering_value based on the image we get
    throttle = 1.0 - speed / maxSpeed # Limit the speed, to not go above the Threshold
    print(f'{steering}, {throttle}, {speed}')
    sendControl(steering, throttle) # Send prediction into the simulator
 
 
@sio.on('connect')
def connect(sid, environ):
    print('Connected')
    sendControl(0, 0) # Sending the command to simulator, Steering, and Speed at the begining 
 
 
def sendControl(steering, throttle):
    sio.emit('steer', data={
        'steering_angle': steering.__str__(),
        'throttle': throttle.__str__()
    })
 
 
if __name__ == '__main__':
    model = load_model('model.h5')
    app = socketio.Middleware(sio, app)
    ### LISTEN TO PORT 4567
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app) # communicate to port 4567