#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 18:38:31 2021

@author: tai
"""

import numpy as np
import os
import pandas as pd
import torch

from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from utils_Yolov3 import (
    iou_width_height as iou,
    non_max_suppression as nms
    )

ImageFile.LOAD_TRUNCATED_IMAGES = True # ?????????????







class YOLODataset(Dataset):
    def __init__(
        self,
        csv_file,
        img_dir,
        label_dir,
        anchors,
        image_size=416,
        S=[13, 26, 52],
        C=20,
        transform=None,
    ):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.transform = transform
        self.S = S
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])  # for all 3 scales
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.C = C
        self.ignore_iou_thresh = 0.5

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist()
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = np.array(Image.open(img_path).convert("RGB"))
        #print("****************************************")
        #print(bboxes)
        #print(label_path)
        #print(img_path)
        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]
           #print(bboxes)

        # Below assumes 3 scale predictions (as paper) and same num of anchors per scale
        targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S]
        for box in bboxes:
            iou_anchors = iou(torch.tensor(box[2:4]), self.anchors)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label = box
            has_anchor = [False] * 3  # each scale should have one anchor
            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                S = self.S[scale_idx]
                i, j = int(S * y), int(S * x)  # which cell
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = S * x - j, S * y - i  # both between [0,1]
                    width_cell, height_cell = (
                        width * S,
                        height * S,
                    )  # can be greater than 1 since it's relative to cell
                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                    )
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True

                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1  # ignore prediction

        return image, tuple(targets)



def test():
    import YOLOv3_config as config
    from utils_Yolov3 import (
    mean_average_precision,
    cells_to_bboxes, # From cell coordinate to original image coordinate
    get_evaluation_bboxes, # mAP, evaluate model
    save_checkpoint,
    load_checkpoint,
    check_class_accuracy, # sometime we check it
    get_loaders, # LONGGGGGGGG 
    plot_couple_examples,
    plot_image
)
    anchors = config.ANCHORS

    transform = config.test_transforms

    dataset = YOLODataset(
        "data/train.csv",
        "data/images/",
        "data/labels/",
        S=[13, 26, 52],
        anchors=anchors,
        transform=transform,
    )
    S = [13, 26, 52]
    scaled_anchors = torch.tensor(anchors) / (
        1 / torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    )
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    for x, y in loader:
        boxes = []

        for i in range(y[0].shape[1]):
            anchor = scaled_anchors[i]
            print(anchor.shape)
            print(y[i].shape)
            boxes += cells_to_bboxes(
                y[i], is_preds=False, S=y[i].shape[2], anchors=anchor
            )[0]
        boxes = nms(boxes, iou_threshold=1, threshold=0.7, box_format="midpoint")
        print(boxes)
        plot_image(x[0].permute(1, 2, 0).to("cpu"), boxes)


if __name__ == "__main__":
    test()

"""
'''
YOLODataset:
    1. [x,y,w,h,confidence,classIdx] -> [x_cell, y_cell, w_cell, h_cell, confidence, classIdx]
    2. Each scale_prediction only have 1 bbox, 
    => we can have 1 -> 3 bboxes_labels for one image, scale_prediction = 1 maybe have more bboxes 
    (because scale between (1,0) or (1,2) have smaller different compare to (0,2))
    
    confidence = 0: no object
    confidence = 1: have object
    confidence = -1: ignore this bboxes
'''
class YOLODataset(Dataset):
    def __init__(
            self,
            csv_file,
            img_dir, label_dir,
            anchors,
            image_size = 416,
            S = [13, 26, 52],
            C = 20,
            transform = None,
            ):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.transform = transform
        self.S = S
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2]) # for all 3 scales
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.C = C
        self.ignore_iou_thresh = 0.5
        
    def __len__(self):
        return len(self.annotations)
    
    '''
    Get image, and, label_bboxes
    label_bboxes -> cell_label_bboxes
    '''
    def __getitem__(self, index):
        # Label Path
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        
        # [class_confidence, x, y, w, h] -> [x, y, w, h, class_confidence] 
        bboxes = np.roll(np.loadtxt(fname = label_path, delimiter = " ", ndmin = 2), 4, axis = 1).tolist() # Watch video for augmentation
        
        # Image Path
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = np.array(Image.open(img_path).convert("RGB")) # Make sure RGB, and np.array
        
        if self.transform:
            augmentations = self.transform(image = image, bboxes = bboxes)
            # Torch Vision only for support classification
            
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]
        
        '''
        3xSxSx6
        num_anchors // 3: ??????
        we assume that we have 3 scale prediction, and, assume that a same number at each cell
        -> num_anchors = 9 (at each cells we have 3 bboxes, and we have 3 scale prediction)
        
        ==> 3x13x13x6 + 3x26x26x6 + 3x52x52x6 = 63882 bboxes for each image prediction
        
        6 : [confidence, x, y, w, h, classIdx]
        '''
        # targets [scale_idx, anchor_on_scale, S, S, 6]
        targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S] # [confidence, x, y, w, h, classIdx]
        
        for box in bboxes:
            # Compare each label_bbox to all anchor
            iou_anchors = iou(torch.tensor(box[2:4], self.anchors)) # NOTE: iou_width_height
            anchor_indices = iou_anchors.argsort(descending = True, dim = 0) # anchors = 3(scale_prediction) x 3 (bboxes/cell)
            
            x, y, width, height, class_label = box
            
            has_anchor = [False, False, False] # Each scale_prediction only have ONE kind of Anchor
            for anchor_idx in anchor_indices:
                # Find the scale_prediction, and find anchor on that scale
                scale_idx = anchor_idx // self.num_anchors_per_scale # scale_prediction index [0, 1, 2[], ex: 8 // 3 = 2 -> scale_prediction = 2, scale_prediction between [0, 1, 2]
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale # anchor_index [0, 1, 2], ex: 8//3 = 1 -> anchor in cell have index = 1, idx between [0, 1, 2]
                
                # Find S, i, j
                '''
                S : scale_prediction
                i, j: cell_idx in that scale_prediction
                '''
                S = self.S[scale_idx] # S : [13, 26, 52]
                i, j = int(S * y), int(S * x) # x = 0.5, S = 13 --> int(6.5) = 6

                anchor_taken = targets[scale_idx][anchor_on_scale, i , j, 0] # targets : [3, 3, S, S, 0] -> confidence_object
                
                
                # Check in rarely case, when 2 bbox in the same scale_prediction, IOU nearly the same
                '''
                not has_anchor[scale_idx] : 1 bbox in one scale_prediction
                not anchor_taken :  ?????? cannot happend
                '''
                if not anchor_taken and not has_anchor[scale_idx]: 
                    
                    # Take the anchor
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1 
                    
                    # Find the relative coordinate to the cells.
                    '''
                    i, j : cells index, we need to know scale prediction S to find i, j
                    => we have found S, i, j above :)) 
                    '''
                    x_cell, y_cell = (S * x - j), (S * y - i) # ex 6.5 - 6 = 0.5 both are between [0, 1]
                    width_cell, height_cell = (
                        width * S, # S = 13, width = 0.5 -> 6.5
                        height * S
                    )
                    
                    # Save Coordinate into targets_label
                    box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                    
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True
                    
                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1 # ignore this prediction
                '''
                NOTE: Những trường hợp khác targets[scale_idx][anchor_on_scale, i, j, 0] = 0 thì bị loại từ đầu rồi 
                '''
                    
        return image, tuple(targets)
                    
   """                 
                    
                    
                
        
        
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        