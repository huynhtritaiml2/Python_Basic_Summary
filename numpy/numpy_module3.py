import numpy as np

x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]])

z = x + y
print(z)
'''
[[ 6  8]
 [10 12]]
'''
z = np.add(x,y)
print(z)
'''
[[ 6  8]
 [10 12]]
'''

print(x-y)
'''
[[-4 -4]
 [-4 -4]]
'''
print(np.subtract(x, y))
'''
[[-4 -4]
 [-4 -4]]
'''

print(x * y)
'''
[[ 5 12]
 [21 32]]

NOTE: Elements wise 
'''

print(np.multiply(x, y))
'''
[[ 5 12]
 [21 32]]
'''

print(x / y)
'''
[[0.2        0.33333333]
 [0.42857143 0.5       ]]
'''

print(np.sqrt(x))
'''
[[1.         1.41421356]
 [1.73205081 2.        ]]
'''

# Dot product == inner product
v = np.array([9, 10])
w = np.array([11, 13])
v.dot(w)
print(v.dot(w)) # 229
print(np.dot(v, w)) # 229


print(x)
'''
[[1 2]
 [3 4]]
'''
print(x.T)
'''
[[1 3]
 [2 4]]
'''

print(np.sum(x)) # 10 
print(np.sum(x, axis = 0))
'''
[4 6]
'''
print(np.sum(x, axis = 1))
'''
[3 7]
'''