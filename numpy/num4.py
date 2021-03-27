# Before we have numpy
a = [1, 2, 3, 4, 5]
b = [10, 11, 12, 13, 14]
# When we try to add 2 list together
print(a + b) # [1, 2, 3, 4, 5, 10, 11, 12, 13, 14]

# SOLUTION:
result = []
for first, second in zip(a, b):
    result.append(first + second)
'''
# NOTE: not difficult but to annoy, and if we have more dimensions in list
And for loop do step by step, rather than parallel, so it slower than what computer can do
'''


import numpy as np

a = np.array([1, 2, 3, 4, 5])
print(type(a)) # <class 'numpy.ndarray'> # ndarray == N dimensions array
print(a.dtype) # int64, so special thing about numpy is all data is int64, not a mess stuff in list
# int type, and 64 bits to present it

f = np.array([1.2, 2.3, 3.4, 4.5, 5.6])
print(f.dtype)  # float64

# Access array
print(a[0])
a[0] = 10 # change value
print(a) # [10  2  3  4  5]

# Chang value, but cannot chang type
a[0] = 11.5 
print(a) # [11  2  3  4  5]


print(a.ndim) # 1 
print(a.shape) # (5,)
print(a.size) # 5 # total number in array, not shape

# Element wise by numpy
a = np.array([10, 11, 12, 13, 14])
f  = np.array([1, 2, 3, 4, 5])
print(a + f)
print(a - f)
print(a * f)
print(a ** f)
print(a / f)
print(a // f)
'''
[11 13 15 17 19]
[9 9 9 9 9]
[10 22 36 52 70]
[    10    121   1728  28561 537824]
[10.    5.5   4.    3.25  2.8 ]
[10  5  4  3  2]
'''
# NOTE: Vectorized Operation, write statement without using loop, above is vectorized operation

# Operation with constant
print( a * 10) # [100  20  30  40  50]
print( a + 10)
print( a - 10)
print( a ** 10)
print( a / 10)
print( a // 10)
'''
[100 110 120 130 140]
[20 21 22 23 24]
[0 1 2 3 4]
[ 10000000000  25937424601  61917364224 137858491849 289254654976]
[1.  1.1 1.2 1.3 1.4]
[1 1 1 1 1]
'''

# Universal function (ufunc)
# Mathmatical function
print(np.sin(a)) # [-0.54402111 -0.99999021 -0.53657292  0.42016704  0.99060736]

# 2 Demensional array
a = np.array([[0, 1, 2, 3],
                [10, 11, 12, 13]])
print(a)
'''
[[ 0  1  2  3]
 [10 11 12 13]]
'''

print(a.shape) # (2, 4)
print(a.size) # 8: 2 * 4 = 8 elements
print(a.ndim) # 2: mean 2 dimesnion

print(a[1, 3]) #13
a[1, 3] = -1
print(a)
'''
[[ 0  1  2  3]
 [10 11 12 -1]]
'''

# Access by row, first dimenstion in 2d array
print(a[1]) # [10 11 12 -1]

# different to MATLAB :))

# SLICING
# var[lower:upper:step], lower bound is included,upper bound is excluded, similar to list
a = np.array([10, 11, 12, 13])
print(a[1:3]) # [11 12], 2 elment from the begining 3 - 1 = 2
print(a[1:-2]) # [11 12], do not know how many element in this, this is make image smaller

# grab first three elements
print(a[:3]) # [10 11 12]: 3 - 0 = 3 elements
# grab last two elements
print(a[-2:]) # [12 13]
# down sampling number element in array
print(a[::2]) # [10 12]

# ARRAY SLICING
a = np.array([  [0, 1, 2, 3, 4, 5], 
                [10, 11, 12, 13, 14, 15],
                [20, 21, 22, 23, 24, 25],
                [30, 31, 32, 33, 34, 35],
                [40, 41, 42, 43, 44, 45],
                [50, 51, 52, 53, 54, 55]])
print(a)
'''
[[ 0  1  2  3  4  5]
 [10 11 12 13 14 15]
 [20 21 22 23 24 25]
 [30 31 32 33 34 35]
 [40 41 42 43 44 45]
 [50 51 52 53 54 55]]
'''
print(a[0, 3:5])
'''
[3 4]
'''
print(a[4: , 4:])
'''
[[44 45]
 [54 55]]
'''
print(a[:, 2]) # [ 2 12 22 32 42 52]

print(a[2::2, ::2])
'''
[[20 22 24]
 [40 42 44]]
'''

############# 
a = np.arange(25)
print(a.dtype) # int64
print(a.shape) # (25,)
print(a.size) # 25
print(a.ndim) # 1

print(a) # [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24]

a = np.arange(25).reshape(5, 5)
print(a.dtype) # int64
print(a.shape) # (5, 5)
print(a.size) # 25
print(a.ndim) # 2

print(a) 
'''
[[ 0  1  2  3  4]
 [ 5  6  7  8  9]
 [10 11 12 13 14]
 [15 16 17 18 19]
 [20 21 22 23 24]]
'''

print(a[1, 0]) # 5 # 
print(a[3, 0]) # 15
print(a[1, 2]) # 7
print(a[3, 2]) # 17
print(a[ : , 1]) # [ 1  6 11 16 21]
print(a[ : , 3]) # [ 3  8 13 18 23]
print(a[ 4 , :]) # [20 21 22 23 24]
print(a[1::2 , :3:2]) # Blue
'''
[[ 5  7]
 [15 17]]
'''
print(a[:, 1::2]) # Red
'''
[[ 1  3]
 [ 6  8]
 [11 13]
 [16 18]
 [21 23]]
'''

print(a[4:]) # [[20 21 22 23 24]]
print(a[-1:]) # [[20 21 22 23 24]] # show the last row

# COPY ARRAY
# NOTE: both a and b share the same memory location
a = np.array([1, 2, 3, 4])
b = a[:2]
b[0] = -1
print(a) # [-1  2  3  4]

# SOLUTION
a = np.array([1, 2, 3, 4])
b = a[:2].copy()
b[0] = -1
print(a) # [1 2 3 4]

# Replace with slicing
a = np.array([0, 1, 2, 3, 4])
# CASE 1: the same size
a[-2:] = [-1, -2]
print(a) # [ 0  1  2 -1 -2]
# CASE2: scalar value ***************************************
a[-2:] = 99
print(a) # [ 0  1  2 99 99]


# parenthesis and square bracket
'''
a[0] is mean __get__item(0)
a[0] = 100 mean __set__item(0, 100)

dunder methods, with the name have double underscore
ex: 
READ this book to master python:
fluent python pdf
https://www.google.com/search?q=fluent+python+pdf&source=lmns&tbm=vid&bih=530&biw=1097&client=ubuntu&hs=Igz&hl=vi&sa=X&ved=2ahUKEwjUz9nFqpXvAhVOTPUHHZUgAicQ_AUoAnoECAEQAg

'''
# Fancy Indexing 
# NOTE: it is useful if we know exactly position
### INDEXING by Position
a = np.arange(0, 80, 10)
indices = [1, 2, -3]
y = a[indices]
print(a) # [ 0 10 20 30 40 50 60 70]
print(y) # [10 20 50]

### Indexing with Booleans
mask = np.array([0, 1, 1, 0, 0, 1, 0, 0], dtype = bool) # buid the mask by hand 
y = a[mask]
print(y) # [10 20 50]

# HOW TO CREATE A MASK
a = np.array([-1, -3, 1, 4, -6, 9, 3])
mask = a < 0
print(mask) # [ True  True False False  True False False]
negative = a[mask]
print(negative) # [-1 -3 -6]
# or Smaller zero = 0
a[mask] = 0
print(a) # [0 0 1 4 0 9 3]


# Fancy indexing in 2D
a = np.array([  [0, 1, 2, 3, 4, 5], 
                [10, 11, 12, 13, 14, 15],
                [20, 21, 22, 23, 24, 25],
                [30, 31, 32, 33, 34, 35],
                [40, 41, 42, 43, 44, 45],
                [50, 51, 52, 53, 54, 55]])
print(a)
'''
[[ 0  1  2  3  4  5]
 [10 11 12 13 14 15]
 [20 21 22 23 24 25]
 [30 31 32 33 34 35]
 [40 41 42 43 44 45]
 [50 51 52 53 54 55]]
'''
# CASE 1:
# Diagonal 
diagonal = [[0, 1, 2, 3, 4], 
            [1, 2, 3, 4, 5]] 
print(tuple(diagonal)) # ([0, 1, 2, 3, 4], [1, 2, 3, 4, 5])
print(a[diagonal]) # Maybe error later
print(a[tuple(diagonal)]) # For future work # [ 1 12 23 34 45]

# CASE 2:
#
print(a[3: , [0, 2, 5]]) # NOTE: because different step size, so we use list to take column we want
'''
[[30 32 35]
 [40 42 45]
 [50 52 55]]
'''
# CASE 3:
#
mask = np.array([1, 0, 1, 0, 0, 1], dtype=bool)
print(a[mask, 2]) # [ 2 22 52]

# Exercise:
a = np.arange(25).reshape(5,5)
mask = [[3, 0, 2, 3],[1, 2, 3, 4]]
print(a[mask]) # [16  2 13 19]

# 
print(0 % 3 ) # 0
divide_by_3 = a[a%3] # WRONG
print(divide_by_3)
'''
[[[ 0  1  2  3  4]
  [ 5  6  7  8  9]
  [10 11 12 13 14]
  [ 0  1  2  3  4]
  [ 5  6  7  8  9]]

 [[10 11 12 13 14]
  [ 0  1  2  3  4]
  [ 5  6  7  8  9]
  [10 11 12 13 14]
  [ 0  1  2  3  4]]

 [[ 5  6  7  8  9]
  [10 11 12 13 14]
  [ 0  1  2  3  4]
  [ 5  6  7  8  9]
  [10 11 12 13 14]]

 [[ 0  1  2  3  4]
  [ 5  6  7  8  9]
  [10 11 12 13 14]
  [ 0  1  2  3  4]
  [ 5  6  7  8  9]]

 [[10 11 12 13 14]
  [ 0  1  2  3  4]
  [ 5  6  7  8  9]
  [10 11 12 13 14]
  [ 0  1  2  3  4]]]
WRONG
'''
# SOLUTION:
divide_by_3 = a[a%3 == 0] # RIGHT
mask = a%3 == 0
print(mask) 
'''
[[ True False False  True False]
 [False  True False False  True]
 [False False  True False False]
 [ True False False  True False]
 [False  True False False  True]]
'''
print(divide_by_3) # [ 0  3  6  9 12 15 18 21 24]


