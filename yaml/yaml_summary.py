#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  8 12:47:32 2021

@author: tai
"""
'''
settings:
  classes: ['motorcycle', 'car', 'bus', 'truck']
  size: [1280,720]
  checkpoint: "./models/deepsort/deep/checkpoint/ckpt.t7"
  cam:
    cam_01: 
      video: '/content/drive/MyDrive/AI_Deep_Learning/Vehicle_counting/cam_01.mp4'
      boxes: '/content/drive/My Drive/results/AIC HCMC 2020/boxes/cam_01'
      zone: '/content/drive/MyDrive/AI_Deep_Learning/Vehicle_counting/cam_01.json'
      tracking_config:
        MAX_DIST: 0.3
        MIN_CONFIDENCE: 0.5
        NMS_MAX_OVERLAP: 0.9
        MAX_IOU_DISTANCE: 0.8
        MAX_AGE: 40
        N_INIT: 4
        NN_BUDGET: 60
'''
import yaml

path = "./cam_configs.yaml"

# Method 1:
yaml_file = open(path)
#print(yaml_file)


attr = yaml.load(yaml_file, Loader = yaml.FullLoader)["settings"]
print(attr["classes"])
print(attr["cam"]["cam_01"]["video"])



# Method 2:
class Config():
    def __init__(self, yaml_path):
        # Method 1:
        #'''
        yaml_file = open(yaml_path)
        #self._attr = yaml.load(yaml_file, Loader = yaml.FullLoader)["settings"]
        self._attr = yaml.load(yaml_file, Loader = yaml.FullLoader)
        #'''
        
        # Method 2: WRONG
        #with open(yaml_path, "r") as faml_file:
        #    self._attr = yaml.load(yaml_file, Loader = yaml.FullLoader)
            
            
    def __getattr__(self, attr):
        #print("*************************")
        #print(attr)
        try:
            return self._attr[attr]
        except KeyError:
            return None
        
yaml_object = Config(path)
data_dict = yaml_object.settings # This return a dictionary, not object *******************
# data_point1 = data_point1.cam_01 # EROOR : 'dict' object has no attribute 'cam_01'
print(yaml_object)
print(data_dict)


data_point2 = yaml_object.settings.get("cam")
data_point3 = data_point2["cam_01"]



print(data_point2)
print(data_point3)

# Write Yaml
dict_data = [{'sports' : ['soccer', 'football', 'basketball', 'cricket', 'hockey', 'table tennis']},
{'countries' : ['Pakistan', 'USA', 'India', 'China', 'Germany', 'France', 'Spain']}]
print("***********Write Yaml**************")
with open("output.yaml", 'w') as f:
    yaml.dump(dict_data, f)
'''
- sports:
  - soccer
  - football
  - basketball
  - cricket
  - hockey
  - table tennis
- countries:
  - Pakistan
  - USA
  - India
  - China
  - Germany
  - France
  - Spain

'''
