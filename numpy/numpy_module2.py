import numpy as np

x = np.zeros((2,3))
print(x)
'''
[[0. 0. 0.]
 [0. 0. 0.]]

dtype : float in default
'''
x = np.zeros((2,3), dtype=int)
print(x)
'''
[[0 0 0]
 [0 0 0]]

'''

x = np.ones((4,5))

print(x)
'''
[[1. 1. 1. 1. 1.]
 [1. 1. 1. 1. 1.]
 [1. 1. 1. 1. 1.]
 [1. 1. 1. 1. 1.]]
'''

x = np.ones((4,5,3))
print(x)
'''
[[[1. 1. 1.]
  [1. 1. 1.]
  [1. 1. 1.]
  [1. 1. 1.]
  [1. 1. 1.]]

 [[1. 1. 1.]
  [1. 1. 1.]
  [1. 1. 1.]
  [1. 1. 1.]
  [1. 1. 1.]]

 [[1. 1. 1.]
  [1. 1. 1.]
  [1. 1. 1.]
  [1. 1. 1.]
  [1. 1. 1.]]

 [[1. 1. 1.]
  [1. 1. 1.]
  [1. 1. 1.]
  [1. 1. 1.]
  [1. 1. 1.]]]
'''

x = np.arange(10)
print(x) # [0 1 2 3 4 5 6 7 8 9] # NOTE: not include 10
print(type(x)) # <class 'numpy.ndarray'>
li = range(10)
print(li) # range(0, 10) :)) FUCK 
print(type(li)) # <class 'range'> # 

x = np.arange(5, 10)
print(x) # [5 6 7 8 9]
x = np.arange(5, 10, 2)
print(x) # [5 7 9]

x = np.linspace(1., 4., 6)
# Start point, End point, Number_output
print(x) # [1.  1.6 2.2 2.8 3.4 4. ]

x = np.full((2,2), 8)
print(x)
'''
[[8 8]
 [8 8]
'''

# Identity matrix
x = np.eye(5)
print(x)
'''
[[1. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0.]
 [0. 0. 1. 0. 0.]
 [0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 1.]]
'''

x = np.random.random((4, 5))
print(x)
'''
[[0.28815614 0.74488132 0.6480484  0.98451944 0.77241998]
 [0.81993297 0.58902555 0.20340241 0.76090825 0.80086367]
 [0.94196975 0.86186953 0.22968986 0.02968917 0.14980167]
 [0.76895804 0.42989449 0.81850894 0.12170297 0.92140965]]
'''


