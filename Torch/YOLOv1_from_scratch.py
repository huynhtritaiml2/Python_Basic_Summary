#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 10:37:47 2021

@author: tai
"""

'''
YOLOv1 train on  PASCAL VOC Dataset
20 classes

Nowsday, COCO Dataset

1. Split into SxS grid or cells, (paper, S = 7)
2. 
Each cell for only predict 1 class, predict 1 midpoint
x = width
y = height

from 0->1

Each ouput and label will be relative to the cell
Each bounding box for each cell will have: [x1, y1, x2, y2]

x1, y1: object midpoint  int the cell
    from 0 -> 1
w, h: 
    maybe greater than 1, w, h maybe larger than the cell

Ex: [0.95, 0.55, 0.5, 1.5]

label_cell = [c1, c2, ..., c20, p_c, x, y, w, h]
c: 1 or 0 for each class
p_c: 1 or 0 if there is a object in cell
x,y,w,h : coordinate of the object


-- Prediction:
look similar to the label_cell, but we we will output two bounding boxes
* Tall Bouding Box, and, Wide Bouding box

pred_cell = [c1, c2, ... c20, p_c1, x1, y1, w1, h1, p_c2, x2, y2, w2, h2]
c1, c2, ... c20: probability of each class
p_c1: probability there is a object in cell 1
p_c2: probability there is a object in cell 2
x1, y1, w1, h1: coordinate box 1
x2, y2, w2, h2: coordinate box 2

NOTE: Limitation of YOLOv1, Only detect one object in each cell

-- Target shape for one images: (S, S, 25)
-- Prediction shape for one images: (S, S, 30)

'''

import torch
import torch.nn as nn

architecture_config = [
    # Tupple: (kernel_size, num_filters, stride, padding)
    (7, 64, 2, 3), # 3: mpadding calculated by hand ***????*??????
    "M",
    (3, 192, 1, 1), 
    "M",
    (1, 128, 1, 0), # Do not need padding for 1x1 kernel
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    
    "M",
    # List: (tupple and the last integer represent number of repeats)
    [(1, 256, 1, 0), (3, 512, 1, 1), 4], # x4
    (1, 512, 1, 0),
    (3, 1024, 1, 1,),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2], # x2 
    (3, 1024, 1, 1),
    (3, 1024, 2, 1), # stride = 2
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]
'''
Need Fully connected layer and reshape in to  (7, 7, 30)

'''

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias = False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))
    
class Yolov1(nn.Module):
    def __init__(self, in_channels = 3, **kwargs):
        super(Yolov1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)
        
    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim =1))
    
    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels
        
        for x in architecture:
            if type(x) == tuple:
                layers += [
                    CNNBlock(in_channels,
                                   out_channels = x[1],
                                   kernel_size = x[0],
                                   stride = x[2],
                                   padding = x[3])
                    ]
                
                in_channels = x[1]
        
            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2))]

            elif type(x) == list:
                conv1 = x[0] # Tupple
                conv2 = x[1] # Tupple
                num_repeats = x[2] # Integer
                
                for _ in range(num_repeats):
                    layers += [
                        CNNBlock(
                            in_channels,
                            conv1[1],
                            kernel_size = conv1[0],
                            stride = conv1[2],
                            padding = conv1[3],)
                        ]
                    layers += [   
                        CNNBlock(
                            conv1[1], # Input_channels
                            conv2[1],
                            kernel_size = conv2[0],
                            stride = conv2[2],
                            padding = conv2[3],)
                        ]

                    in_channels = conv2[1]
            
        return nn.Sequential(*layers)

    def _create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 496), # Original paper this should be 4096, because it take to much RAM
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(496, S * S * (C + B * 5)), # (7,7, 30)
            )

def test(S = 7, B = 2, C = 20):
    model = Yolov1(split_size = S, num_boxes = B, num_classes = C)
    x = torch.rand((2, 3, 448, 448))
    print(model(x).shape)

if __name__ == "__main__":    
    test()


'''
Loss Function:

** Identiry function: 
- In every cell:
- In every bouding boxes:
If confidence > threshold:
    (x_label - x_pred)^2 + (y_label - y_pred)^2

Lambda_coord = 5 , priorite for this error

** w, h loss:
we have square root, to for the case too big bboxes

Lambda_coord = 5 , priorite for this error

** Confidence error:
(C_label - C_pred)^2

** Not object in the cell
(C_label - C_pred)^2

Lambda_no_obj = 0.5

** Classification error:
For each cell
For each Class
p((c)_label - p(c)_pred)^2 

NOTE: Cross Enotropy Loss, Negative Log likehood, But this case use Regression


'''


























