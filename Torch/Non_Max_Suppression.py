#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 12:27:03 2021

@author: tai
"""

'''
suppression: sự đàn áp

1. Discarding all bounding boxes < probability threshold
2.
While BoudingBoces:
    - Take out the largest probability box
    - Remove all other boxes with IoU > threshold_IoU
    
    (And we do it for all classes)
'''
import torch
from IOU import intersection_over_union

#def non_max_suppression(
def nms(
        bboxes,
        iou_threshold,
        prob_threshold,
        box_format="corners" # depend on what kind of bbox
        ):
    # bboxes = [[1, 0.9, x1, y1, x2, y2]]
    assert type(bboxes) == list
    
    # Step 1:
    bboxes = [box for box in bboxes if box[1] > prob_threshold] 
    # Step 2:
    # - Take out the largest probability box
    bboxes = sorted(bboxes, key = lambda x: x[1], reverse = True)
    bboxes_after_nms = []
    
    while bboxes:
        chosen_box = bboxes.pop(0)
        
        '''
        chosen_box: will discard all localization if they are the same class and have IoU < 0.5, ex
        bboxes: will decrease the size after each loop, 
                because it process each class at the time.
                
        Note: 2 the same class maybe in the bboxes_after_nms
        '''
        bboxes = [
            box 
            for box in bboxes 
            if box[0] != chosen_box[0] 
            or intersection_over_union(torch.tensor(chosen_box[2:]),
                                       torch.tensor(box[2:]), 
                                       box_format = box_format) < iou_threshold
                                                    
            ]
        bboxes_after_nms.append(chosen_box)
    return bboxes_after_nms
    


