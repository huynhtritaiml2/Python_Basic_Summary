# Multi-Dimentsional Arrays
# Image: READ IT
# https://www.google.com/search?q=visualizing+multidimensional+array&client=ubuntu&hs=Sqg&sxsrf=ALeKk02toeLHpZCojxdrR2Nhw_IMdK9vjQ:1614823068568&source=lnms&tbm=isch&sa=X&ved=2ahUKEwjByoHbxJXvAhVF7WEKHbf1D0AQ_AUoAXoECAMQAw&biw=1097&bih=530#imgrc=IeU5JN9Yz9QJTM
import numpy as np
# Different in MATLAB :)) NOTE IT 


# Cmputations with Arrays
'''
Rule 1: Operation between multiple array objects are first checked for proper shape match
Dimension Broadcasting/ Array Broadcasting
Rule 2:
Mathematical operator ( + - * / exp log) apply element by element
Rule 3:
Reduction operation (mean, std, skew, kurt, sum, prod, ...) apply to the whole array, unless an axis is specified
Rule 4:
Missing vale propagate unless explicitly ignored (nanmean, nansum)
np.nan + 1 = nan # nan = Not a Number

'''
# Rule 1: Smaller array is broadcasted to math larer array
'''
Rule 1.1 : if two array different in their dimension, the shape of one with fewer dimensions is padded with 1 on its leading side (left side)
-- If two array have the same size:
Rule 1.2 : if two array do not math in any dimension. the array with shape equal to 1 is stretch to math the other shape
Ryle 1.3 : If in nay dimension the size disagree and NOT equal to 1, an error is raised

https://www.youtube.com/watch?v=0u9OzBSRZec&list=PLzgPDYo_3xukqLLjNeuCxj4CwvkJin03Z&index=19
a (5, 3)
b (3, )

a + b 
a   (5, 3)
b   (1, 3)

a (5, 1)
b (3, )

a + b 
a   (5, 1)
b   (1, 3)
'''
a = np.ones((3, 5))
print(a.shape) # (3, 5)
b = np.ones((5)) # (5,)
b = np.ones(5) # (5,)
b = np.ones((5,)) # (5,)
# b = np.ones((5,1)) # WRONG (5, 1) NOT (1 , 5)
# b = np.ones((1, 5)) # (1, 5)
print(b.shape) 

b.reshape(1, 5)
print(b.shape) # 
b = np.ones((5,))
b[np.newaxis, : ]
print(b.shape)

a = np.array([[1, 2], [3, 4], [5, 6]])
print(a) # (3, 2)
print(a.shape)
'''
[[1 2]
 [3 4]
 [5 6]]
'''

b = np.array([10, 20])
print(b.shape) # (2,)

s = a + b
print(s)
print(s.shape)
'''
[[11 22]
 [13 24]
 [15 26]]

(3, 2)
'''

c = np.array([10, 20, 30])
#s = a + c # ERROR
'''
# ValueError: operands could not be broadcast together with shapes (3,2) (3,) 
Because 
a (3, 2)
c (1 , 3)
2 != 3 and do not have any dimension = 1, so raise error
ERROR RULE 1.3
'''

b = np.array([[10, 20],[30, 40]])
#s = a + b# ERROR
'''
ValueError: operands could not be broadcast together with shapes (3,2) (2,2) 


a (3, 2)
c (2 , 2)

consider from right to left:

2 == 2: 
then, consider next dimension

3 != 2 and do not have any dimension = 1, so raise error
ERROR RULE 1.3

'''

a = np.array([1, 2, 3, 4, 5]).reshape(5,1)
b = np.array([10, 11, 12]).reshape(1, 3)
s = a + b
print(s)
'''
[[11 12 13]
 [12 13 14]
 [13 14 15]
 [14 15 16]
 [15 16 17]]

a (5,1)
b (1, 3)

1 != 3, but we have 1
5 != 1, but we have 1
'''



### ARRAY CALCULATION METHODS #####################################
'''
# NOTE: dimension we choose will go away.


'''
a = np.array([[1, 2, 3], [4, 5, 6]])
print(a)
'''
(2 ,3)
[[1 2 3]
 [4 5 6]]
'''
print(a.sum()) # 21
print(a.sum(axis=0)) # [5 7 9]
print(a.sum(axis=-1)) # [ 6 15] ****************

# METHOD 2: for sum
print(np.sum(a)) # 21 
print(np.sum(a, axis = 0)) # [5 7 9]

# Min/ Max
a = np.array([[2, 3], [0, 1]])

# min/max return value, NOT index
print(np.min(a)) # 0
print(np.max(a)) # 3

print(np.max(a, axis = 0)) # [2 3]

# argmax return index of max/min, after turn maxtrix into 1 D array
print(a.argmax()) #  1
print(a.argmin()) #  2

# unreval_index give us position in matrix, better than above only 1D array
print(np.unravel_index(a.argmax(), a.shape)) # (0, 1)
print(np.unravel_index(a.argmin(), a.shape)) # (1, 0)

# np.where, find index of value of element
a = np.array([-1, 2, 5, 5, 3])
print(a == a.max()) # [False False  True  True False] **************
mask = a == a.max()
where = np.where(a == a.max())
print(where) # (array([2, 3]),) at position 2 and 3
where = np.where(a > 0)
print(where) # (array([1, 2, 3, 4]),) at position 1 2 3 4

