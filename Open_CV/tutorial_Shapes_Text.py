#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 20:54:29 2021

@author: tai
"""

import cv2
import numpy as np

'''
************************** Create a image for Opencv **************************
'''
img = np.zeros((512, 512)) # Black image with only 1 channel
img = np.zeros((512, 512, 3))
#print(img.shape) # (512, 512, 3)
#print(img.dtype) # float64, but we need integer

img = np.zeros((512, 512, 3), dtype = np.uint8)
#print(img.dtype) # uint8
img[:] = 255, 0, 0 # it will broadcast image into Blue image

######## Change the blue image in the certain region
img = np.zeros((512, 512, 3), dtype = np.uint8)
img[20:30, 60:100] = 255, 0, 0 # it will broadcast image into Blue image, But in smaller region

'''
*********** So this is the method we can draw everything, Ex: a line, a circle *************
'''
img = np.zeros((512, 512, 3), dtype = np.uint8)
''' 
1. Line:
cv2.line(img, starting_point, endding_point, color, thickness)'''
#cv2.line(img, (0, 0), (100, 100), (0, 255, 0), 2)
cv2.line(img, (0, 0), (img.shape[1], img.shape[0]), (0, 255, 0), 2)

''' 
2. Rectangle
cv2.rectangle(img, starting_point, endding_point, color, thickness)'''
#cv2.rectangle(img, (350, 100), (450, 200), (0, 0, 255), 2) # Not fill, thicness = 2
cv2.rectangle(img, (350, 100), (450, 200), (0, 0, 255), cv2.FILLED)

''' 
3. Circle:
cv2.circle(img, middle_point, radius , color, thickness)'''
#cv2.circle(img, (150, 400), 50, (255, 0, 0), 3) # Not fill, thicness = 3
cv2.circle(img, (150, 400), 50, (255, 0, 0), cv2.FILLED)


''' 
4. put Text:
cv2.putText(img, message_string, starting_point , font, scale, color, thickness)'''
cv2.putText(img, "Draw Shapes", (75, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 150, 0), 1)



cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()