#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 16:46:48 2021

@author: tai
"""

import torch 
import os
import pandas as pd
from PIL import Image
'''
Loading custom data in pytorch
annotations: chú thích


train.csv
img_dir     label_dir
000005.jpg	000005.txt
000007.jpg	000007.txt
000009.jpg	000009.txt
000016.jpg	000016.txt

Label_bbox:
000001.txt
11 0.34419263456090654 0.611 0.4164305949008499 0.262
14 0.509915014164306 0.51 0.9745042492917847 0.972


VOCDataset: Load 1 example from 
'''
class VOCDataset(torch.utils.data.Dataset): # What is torch.utils.data.Dataset ????????????
    def __init__(self, csv_file, img_dir, label_dir, S = 7, B = 2, C = 20, transform = None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C
        
    def __len__(self):
        return len(self.annotations)
    
    '''
    Return: img, and label_matrix
    '''
    def __getitem__(self, index):
        # Read Label_file
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = []
        
        with open(label_path) as f:
            for label in f.readlines():
                '''
                The first index/column is integer
                the x, y, width, height is float so we keep it
                '''
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                    ]
            
            boxes.append([class_label, x, y, width, height])
        
        # Load Image_file
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path) # RGB image
        boxes = torch.tensor(boxes) # Transform to Tensor 
        
        if self.transform: # Maybe add more for augmenting images.
            image, boxes = self.transform(image, boxes)
        
        # Create label_matrix for each image    
        # (S, S, 30)
        label_matrix = torch.zeros((self.S, self.S, self.C + 5*self.B))
        for box in boxes: # more than one bbox in image.
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)
            
            # Find coordinate for the S cells:
            # Transform image coordinate to cell coordinate, relative to cell rather than entire image
            '''
            x_cell, y_cell : is ratio for smaller cell
            x_cell = self.S * x - int(self.S * x) : Lấy phần dư của số lẻ
            '''
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i
            
            '''
            width_cell, height_cell:
            width_cell = width * self.S # 
            Ví dụ, width = 0.6 image_width, mà image_width = 7*cell_width
            -> width = 0.6 * (7*cell_width) =  0.6 * 7
            '''
            width_cell, height_cell = (
                width * self.S,
                height * self.S,
                )
            
            
            if label_matrix[i, j, 20] == 0: # ????????? 100% is zero :)) FUCKING 
                # p(c) : confidence 
                label_matrix[i, j, 20] = 1
                
                # (x, y, w, h)
                box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                label_matrix[i, j, 21:25] = box_coordinates 
                
                # 20 classes
                label_matrix[i, j, class_label] = 1 
                    
        return image, label_matrix
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            