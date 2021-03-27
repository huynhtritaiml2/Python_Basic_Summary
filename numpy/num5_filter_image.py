"""
Filter Image
------------
Read in the "dc_metro" image and use an averaging filterto "smooth" the image.  
Use a "5 point stencil" where you average the current pixel with its neighboring pixels::
              0 0 0 0 0 0 0
              0 0 0 x 0 0 0
              0 0 x x x 0 0
              0 0 0 x 0 0 0
              0 0 0 0 0 0 0
Plot the image, the smoothed image, and the difference between the two.
Bonus
~~~~~
Re-filter the image by passing the result image through the filter again. Do this 50 times and plot the resulting image.
See :ref:`filter-image-solution`.
"""

import matplotlib.pyplot as plt

img = plt.imread('dc_metro.png')
print(img.dtype) # float32
print(img.shape)  # (461, 615)
print(img.size) # 283515
print(img.ndim) # 2
print(img)
'''
[[0.6039216  0.6431373  0.6784314  ... 0.59607846 0.5921569  0.63529414]
 [0.6666667  0.654902   0.654902   ... 0.654902   0.64705884 0.6392157 ]
 [0.68235296 0.6784314  0.6666667  ... 0.61960787 0.57254905 0.5764706 ]
 ...
 [0.14509805 0.15294118 0.16078432 ... 0.1254902  0.1254902  0.1254902 ]
 [0.14117648 0.15686275 0.16470589 ... 0.12156863 0.1254902  0.12156863]
 [0.14117648 0.15294118 0.16470589 ... 0.12156863 0.1254902  0.12156863]]
'''
plt.imshow(img, cmap=plt.cm.hot)
plt.show()
