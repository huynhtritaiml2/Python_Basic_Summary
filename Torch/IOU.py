#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 05:27:21 2021

@author: tai
"""
import torch
import numpy as np


def intersection_over_union(boxes_preds, boxes_labels, box_format = "midpoint"):
    # boxes_preds.shape (N, 4) where N is the number of bboxes
    # boxes_labels.shape is (N, 4)
    
    if box_format == "midpoint":
        wp , hp = boxes_preds[..., 2:3], boxes_preds[..., 3:4]
        
        box1_x1 = boxes_preds[..., 0:1] - wp/2
        box1_y1 = boxes_preds[..., 1:2] - hp/2
        box1_x2 = boxes_preds[..., 2:3] + wp/2
        box1_y2 = boxes_preds[..., 3:4] + hp/2
        
        wl , hl = boxes_labels[..., 2:3], boxes_labels[..., 3:4]

        box2_x1 = boxes_labels[..., 0:1] - wl/2
        box2_y1 = boxes_labels[..., 1:2] - hl/2
        box2_x2 = boxes_labels[..., 2:3] + wl/2
        box2_y2 = boxes_labels[..., 3:4] + hl/2
    
    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1] # box1_x1 = boxes_preds[..., 0:1] shape is (N, )
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        
        print(box1_x1.shape)
        
        box2_x1 = boxes_labels[..., 0:1] 
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)
    
    # .clamp(0) is for the case when they do not intersect, >=0
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    #print(torch.Tensor([-1, 0.1, 0, 1, 2]).clamp(0)) # tensor([0.0000, 0.1000, 0.0000, 1.0000, 2.0000])
    
    box1_area = abs((box1_x2 - box1_x1) * (box1_y1 - box1_y2))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y1 - box2_y2))
    
    return intersection / (box1_area + box2_area - intersection + 1e-6)

'''
intersection_over_union(torch.ones((10, 4)), torch.ones((10,1)))
torch.Tensor([])
iou = intersection_over_union
'''




