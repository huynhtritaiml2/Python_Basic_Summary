#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 12:44:18 2021

@author: tai
"""

'''
Evaluate Object Detection Model:
1. Get all bounding box prediction on our test set
Table:
    Image_name  Confidence_value    TP_or_FP
    image1      0.3                 FP
    image3
    
2. Sort by descending confidence score:
    The largest confidence value at top of the table

3. Calculate Precision and Recall:
    Precision: Dự Đoán đúng / (tất cả dự đoán)
    Recall: Dự đoán đúng / (Label_đúng)

NOTE: ???????????????/ Bước này Biết cách làm nhưng không hiểu ý nghĩa
4. Plot the Precision-Recall graph

5. Calculate Area Under PR curve 

Dog AP = 0.533

6. Do all classes

Dog AP = 0.533
Cat AP = 0.74

mAP = (0.533 + 0.74) / 2 = 0.6365

7. Average for IoU_Threshold 
mAP @ 0.5: 0.05 : 0.95
That mean
mAP @ 0.5 = 
mAP @ 0.55 = 
0.6 =
.
.
.
mAP @ 0.95 = 

Sum all of them and take average
'''

import torch
from collection import Counter
from IOU import intersection_over_union

'''
For single Iou_Threshold
For Testing Phase, not training Because, we do not have loss function
'''
def mean_average_precision(
        pred_boxes, true_boxes, iou_threshold=0.5, box_format="corners",
        num_classes = 20):
    
    # pred_boxes (list) : [[train_idx, class_pred, prob_score, x1, y1, x2, y2], ...]
    average_precisions = [] # mAP
    epsilon = 1e-6
    
    for c in range(num_classes): # For each class, Dog and Cat
        detections = []
        ground_truths = []
        
        # Step 1: Get all bounding box prediction on our test set
        # Summarise all Dog bboxs predicted in test set image
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection) # 
        
        # Summarise all Dog bboxs label/ground_truth in test set image
        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)
        
        '''
        Theo dõi trường hợp 2 bbox cho cùng 1 vật thể, Trừng phạt trường hợp bbox có confidence nhỏ hơn
        # img 0 has 3 bboxes
        # img 1 has 5 bboxes
        # amount_bboxes = {0:3 , 1:5}
        '''
        amount_bboxes = Counter([gt[0] for gt in ground_truths])
        
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)  
        # amount_boxes = {0: torch.tensor([0, 0, 0]), 1: torch.tensor([0, 0, 0, 0, 0])}
        # Use this to calculate Precision and Recall
        
        # Step 2: Sort by descending confidence score:
        detection.sort(key = lambda x: x[2], reverse = True)
        
        # Step 3: Calculate Precision and Recall:
        TP = torch.zeros((len(detections))) # Create column for TP and FP
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)
        
        '''
        After sorting, each row have  
        [train_idx, class_pred, prob_score, x1, y1, x2, y2]
        we only use:
        [train_idx, prob_score, x1, y1, x2, y2] 
        1. For each bbox in detection
        compare to all bbox in Ground_Truth
        -> find best_iou
        
        2. best_iou > 0.5 : TP = 1, FP = 0
            best_iou < 0.5 : FP = 1, TP = 0
            
        NOTE: detection in confidence order
        
        Tại đây, Chúng ta sau đi loại bỏ bbox có confidence < confidence_threshold
        và loại bỏ bbox có Non-Max Suppresion, thì vẫn còn trường hợp 2 bbox cùng thể hiện 1 vât thể
        
        Nên, chúng ta chọn bbox với confidence lớn hơn
        Set TP = 1, FP = 0
        và, bbox với confidence nhỏ hơn:
        Set TP = 0, FP = 1 
        
        Summary:  Trừng phạt mAP
        1. một vật thể có 2 bbox (1 trong 2, với confidence nhỏ hơn)
        2. iou of bbox_predict and bbox_label quá thấp 
        
        
        '''
        for detection_idx, detection in enumerate(detections):
            ground_truth_img = [bbox for bbox in ground_truths if bbox[0] == detection[0]]
            
            num_gts = len(ground_truth_img)
            best_iou = 0
            
            '''

            '''
            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]), torch.tensor(gt[3:]),
                    box_format_box = box_format,             
                    )
                # NOTE: pred_boxes (list) : [[train_idx, class_pred, prob_score, x1, y1, x2, y2], ...]
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx
            
            if best_iou > iou_threshold: # Larger than 0.5, TP = 1, FP = 0, ver vice
                # Trường hợp 2 bboxes cho một vật thể, dư bbox sẽ bị trừng phạt 
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1
            else:
                FP[detection_idx] = 1
        
        # [1, 1, 0, 1, 0] -> [1, 2, 2, 3, 3]
        # Tạo phần tử số cho TP và FP 
        TP_cumsum = torch.cumsum(TP, dim = 0)
        FP_cumsum = torch.cumsum(FP, dim = 0)
        
        # Final value for Precisions and Recalls
        '''
        Mẫu số của recalls luôn cố định (= total bbox_label in test set)
        Mẫu số của precision tăng theo số lượng bbox_prediction trong test set
        
        Tử số của recalls và precision tăng theo TP, có giá trị nhu nhau
        '''
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        
        # Step 4. Plot the Precision-Recall graph
        '''
        vì Graph bắt đầu tại
        x = 0 = recall[0]
        y = 1 = precision[0]
        
        Xem trong hình vẽ
        '''
        precisions = torch.cat((torch.tensor[1], precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        
        # Step 5: Calculate Area Under PR curve, Tính diện tích
        average_precisions.append(torch.trapz(precisions, recalls)) # ????????????//
        
    return sum(average_precisions) / len(average_precisions) # (mAP of Dog + mAP of Cat) / 2

        



















