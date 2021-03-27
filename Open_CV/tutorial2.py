import cv2
import random as random

img = cv2.imread("./assets/hoian.jpg", -1)
# Standard RGB
# OpenCv BGR
# 0: black
# 255: white 
'''
image 2x2
[
[[0, 0, 0], [255, 255, 255]]
[[0, 0, 0], [255, 255, 255]]
]

[0, 0, 0] : BRG is a pixel 
'''
#print(img)
'''
[[[255 251 246]
  [255 251 246]
  [255 251 246]
  ...
  [255 255 255]
  [255 255 255]
  [255 255 255]]

 [[255 251 246]
  [255 251 246]
  [255 251 246]
  ...
  [255 255 255]
  [255 255 255]
  [255 255 255]]

 ...

 [[218 212 207]
  [229 223 218]
  [234 225 221]
  ...
  [238 187 114]
  [242 193 119]
  [248 199 123]]]
'''
print(type(img)) # <class 'numpy.ndarray'>
print(img.shape) # (476, 768, 3)
print(img[90][0]) # [117 107 119] is one pixel at row 90, column 0

###############################################
'''
Add random pixel or add noise into the image
'''
for i in range(100):
    for j in range(img.shape[1]): # for row in rows
        # (rows, columns, channels)
        img[i][j] = [random.randint(0, 255), random.randint(0, 255),random.randint(0, 255)]

cv2.imshow("Hoi An",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#####################################33 Replace the image
tag = img[300:400, 300:600]
img[50:150, 200:500] = tag # NOTE: SHOULD BE the same shape :)) Khó vl 

cv2.imshow("Hoi An",img)
cv2.waitKey(0)
cv2.destroyAllWindows()