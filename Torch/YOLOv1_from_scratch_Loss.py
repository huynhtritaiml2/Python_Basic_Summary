#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 13:51:34 2021

@author: tai
"""

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

"""
import torch
import torch.nn as nn

from IOU import intersection_over_union

class YoloLoss(nn.Module):
    def __init__(self, S = 7, B = 2, C = 20):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction = "sum")
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = 5
        self.lambda_noobj = 0.5
        
    def forward(self, predictions, target):
        predictions = predictions.reshape(-1, self.S, self.S , self.C, self.C + self.B * 5) # Reshape into Final Output :))
        # This is not do in the network
        # prediction : (N, S, S, 30)
        
        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25])
        iou_b2 = intersection_over_union(predictions[..., 25:30], target[..., 25:30])
        # iou_b1 : (N, S, S, 1)
        
        # Find the largest bbox between iou_b1 and iou_b2
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0) 
        iou_maxes, best_box = torch.max(ious, dim = 0) # ?????????? Use Only best_box
        
        # Identity Object in cell i, Iobj_i
        # -- Target shape for one images: (S, S, 25)
        exists_box = target[..., 20].unsqueeze(3) # 1 or 0, if there is object in cell
        #exists_box = target[..., 20:21] # Method 2:
        # exists_box: (N, S, S, 1)
            
        # ============================= #
        #       FOR BOX COORDINATES     #
        # ============================= #
        '''
        1. Sau khi predict xong, mỗi cell có 2 bbox_pred
        2. Chọn bbox có confidence lớn nhất, bỏ những cái khác
        3. Nếu bbox_pred có chung vị trí bbox_label thì giữ lại, còn bbox trong cell khác
        bỏ hết, vì có bbox_label đâu mà so sánh :))
    
        NOTE: chắc bbox dư đó dùng tính loss khác , không phải loss này
        '''
        # For Prediction bboxes
        # box_predictions: (N, S, S, 4)
        box_predictions = exists_box * (
            (
                best_box * predictions[..., 26:30]
                + (1 - best_box) * predictions[..., 21:25]
            )
        )
        '''
        Tính căn w, và căn h:
        Problem: w và h có thể âm (vì giá trị khởi tạo ban đầu hoặc vì dùng leakyReLU (hình như vậy) ????
        -> cần dùng torch.abs trước khi torch.sqrt
        Then, mới lấy dấu của w, và h sau :)) , bằng cách dùng torch.sign
    
        +1e-6 vì derivative có thể inf, ???????? Check not sure ????????
        '''
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4])) + 1e-6 
        
        # For Label bboxes
        # box_targets: (N, S, S, 4)
        box_targets = exists_box * target[..., 21:25] # (x, y, w, h)
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4]) # (w, h)
        
        
        # (N, S, S, 4) -> (N*S*S, 4)
        # MSE(x_pred, x_label) == 1/N*(x_label - x_pred)^2
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim = -2),
            torch.flatten(box_targets, end_dim = -2),
            )
        
        # ====================== #
        #    FOR OBJECT LOSS     #
        # ====================== #
        pred_box = (
            best_box * predictions[..., 25:26] 
            + (1 - best_box) * predictions[..., 20:21]) # 25:26 , 20:21 to keep the demension
        
        # (N*S*S, 1) ->(N*S*S) 
        # We do not use end_dim = -2, because they are in the same shape, by default
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box), 
            torch.flatten(exists_box * target[..., 20:21])
        )
        
        # ======================= #
        #    FOR NO OBJECT LOSS   #
        # ======================= #
        # (N, S, S, 1) -> (N, S*S) or (N, S*S*1)
        # Box 1
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim = 1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim = 1)
        )
        # Box 2
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim = 1), # change this line
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim = 1)
        )
        
        # =================== #
        #    FOR CLASS LOSS   #
        # =================== #
        # (N, S, S, 20) - > (N*S*S, 20)
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim = -2),
            torch.flatten(exists_box * target[..., :20], end_dim = -2)
        )
        
        ################## TOTAL LOSS ##################
        loss = (
            self.lambda_coord * box_loss # First two rows of loss in paper
            + object_loss
            + self.lambda_noobj * no_object_loss
            + class_loss
        )
        
        return loss
    
"""

import torch
import torch.nn as nn
from IOU import intersection_over_union


class YoloLoss(nn.Module):
    """
    Calculate the loss for yolo (v1) model
    """

    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")

        """
        S is split size of image (in paper 7),
        B is number of boxes (in paper 2),
        C is number of classes (in paper and VOC dataset is 20),
        """
        self.S = S
        self.B = B
        self.C = C

        # These are from Yolo paper, signifying how much we should
        # pay loss for no object (noobj) and the box coordinates (coord)
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        # predictions are shaped (BATCH_SIZE, S*S(C+B*5) when inputted
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)

        # Calculate IoU for the two predicted bounding boxes with target bbox
        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25])
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)

        # Take the box with highest IoU out of the two prediction
        # Note that bestbox will be indices of 0, 1 for which bbox was best
        iou_maxes, bestbox = torch.max(ious, dim=0)
        exists_box = target[..., 20].unsqueeze(3)  # in paper this is Iobj_i

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #

        # Set boxes with no object in them to 0. We only take out one of the two 
        # predictions, which is the one with highest Iou calculated previously.
        box_predictions = exists_box * (
            (
                bestbox * predictions[..., 26:30]
                + (1 - bestbox) * predictions[..., 21:25]
            )
        )

        box_targets = exists_box * target[..., 21:25]

        # Take sqrt of width, height of boxes to ensure that
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #

        # pred_box is the confidence score for the bbox with highest IoU
        pred_box = (
            bestbox * predictions[..., 25:26] + (1 - bestbox) * predictions[..., 20:21]
        )

        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 20:21]),
        )

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #

        #max_no_obj = torch.max(predictions[..., 20:21], predictions[..., 25:26])
        #no_object_loss = self.mse(
        #    torch.flatten((1 - exists_box) * max_no_obj, start_dim=1),
        #    torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
        #)

        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
        )

        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        )

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #

        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim=-2,),
            torch.flatten(exists_box * target[..., :20], end_dim=-2,),
        )

        loss = (
            self.lambda_coord * box_loss  # first two rows in paper
            + object_loss  # third row in paper
            + self.lambda_noobj * no_object_loss  # forth row
            + class_loss  # fifth row
        )

        return loss
    

        
        
        
        
        
        
        
        