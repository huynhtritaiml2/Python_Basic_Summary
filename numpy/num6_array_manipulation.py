import numpy as np

a = np.arange(10)
b = np.reshape(a, (5,2))
print(b)
'''
[[0 1]
 [2 3]
 [4 5]
 [6 7]
 [8 9]]
'''
b = np.reshape(a, (5,2), order="F")
print(b)
'''
[[0 5]
 [1 6]
 [2 7]
 [3 8]
 [4 9]]
'''

a = np.arange(5)
b = np.resize(a, (2,3))
print(b)
'''
[[0 1 2]
 [3 4 0]]

'''


a = np.arange(5)
b = np.resize(a, (3,3))
print(b)
'''
# This repeat itself
[[0 1 2]
 [3 4 0]
 [1 2 3]]

'''
################################## flatten and ravel ##################################
'''
flatten: return a copy
ravel : return from original, share the same memory
'''
a = np.array([[1,2,3],[4,5,6],[7,8,9]])
b = a.flatten()
print(b) # [1 2 3 4 5 6 7 8 9]
b[0] = 10
print(a)
'''
[[1 2 3]
 [4 5 6]
 [7 8 9]]

The same result
'''

a = np.array([[1,2,3],[4,5,6],[7,8,9]])
b = a.ravel()
print(b) # [1 2 3 4 5 6 7 8 9]
b[0] = 10
print(a)
'''
[[10  2  3]
 [ 4  5  6]
 [ 7  8  9]]

Change b will change a
'''

######################################## transpose #################################

a = np.arange(1, 11).reshape(5,2)
print(a)
'''
(5, 2)
[[ 1  2]
 [ 3  4]
 [ 5  6]
 [ 7  8]
 [ 9 10]]
'''
b = np.transpose(a)
print(b)
'''
(2,5)
[[ 1  3  5  7  9]
 [ 2  4  6  8 10]]
'''

# 3-D array
a = np.arange(1, 25).reshape(2,3,4)
print(a)
'''
[[[ 1  2  3  4]
  [ 5  6  7  8]
  [ 9 10 11 12]]

 [[13 14 15 16]
  [17 18 19 20]
  [21 22 23 24]]]
'''

b = np.transpose(a)
print(b)
print(b.shape)
'''
(4, 3, 2)
[[[ 1 13]
  [ 5 17]
  [ 9 21]]

 [[ 2 14]
  [ 6 18]
  [10 22]]

 [[ 3 15]
  [ 7 19]
  [11 23]]

 [[ 4 16]
  [ 8 20]
  [12 24]]]
'''


################################## Swap axis #################################

a = np.arange(1, 25).reshape(2,3,4)
print(a)
'''
[[[ 1  2  3  4]
  [ 5  6  7  8]
  [ 9 10 11 12]]

 [[13 14 15 16]
  [17 18 19 20]
  [21 22 23 24]]]
'''
print(np.swapaxes(a, 1, 2))
'''
Swap row and column
[[[ 1  5  9]
  [ 2  6 10]
  [ 3  7 11]
  [ 4  8 12]]

 [[13 17 21]
  [14 18 22]
  [15 19 23]
  [16 20 24]]]
'''

############################# concatenate ############################
a = np.arange(6)
b = np.arange(10,16)
c = np.arange(20,26)

d = np.concatenate((a,b,c))
print(d) # [ 0  1  2  3  4  5 10 11 12 13 14 15 20 21 22 23 24 25]

a = np.arange(6)
b = np.arange(10,16)
c = np.zeros(12)
d = np.concatenate((a,b), out=c) # save to c and save to d
print(d) # [ 0.  1.  2.  3.  4.  5. 10. 11. 12. 13. 14. 15.]
print(c) # [ 0.  1.  2.  3.  4.  5. 10. 11. 12. 13. 14. 15.]
d[0] = 100
print(d) # [100.   1.   2.   3.   4.   5.  10.  11.  12.  13.  14.  15.]
print(c) # [100.   1.   2.   3.   4.   5.  10.  11.  12.  13.  14.  15.]
#NOTE: this is copy 

# 2-d array, 2 array must in dimension
a = np.array([[1,2],[3,4]])
b = np.array([[5,6]])
c = np.concatenate((a,b))
c = np.concatenate((a,b), axis = 0) # similar
print(c)
'''
[[1 2]
 [3 4]
 [5 6]]
'''

a = np.array([[1,2],[3,4]])
b = np.array([5,6])
#c = np.concatenate((a,b))
'''
ValueError: all the input arrays must have same number of dimensions

'''


a = np.array([[1,2],[3,4]])
b = np.array([[5,6]])
''']
ERROR: if 
c = np.concatenate((a,b), axis = 1) # ValueError: all the input array dimensions except for the concatenation axis must match exactly
'''
# SOLUTION:
c = np.concatenate((a,b.T), axis = 1)
print(c)
'''
[[1 2 5]
 [3 4 6]]
'''

################################# vstack #################################
'''
in 1d
1d -> 2d
in 2-d, 3-d
vstack == concatenate(axis = 0)
'''

a = np.arange(6)
b = np.arange(5)
c = np.arange(5)

d = np.vstack((b,c))
print(d)
'''
[[0 1 2 3 4]
 [0 1 2 3 4]]
'''

#d = np.vstack((a,c))
print(d)
'''
ERROR: ValueError: all the input array dimensions except for the concatenation axis must match exactly
'''

################################# hstack #################################
'''
in 2-d, 3-d, 4-d
hstack == concatenate(axis = 1)
in 1-d: output is 1d
hstack == concatenate(axis = 0)

'''
a = np.arange(6)
b = np.arange(5)
c = np.arange(5)
d = np.hstack((a,b))
print(d) # [0 1 2 3 4 5 0 1 2 3 4]
d = np.hstack((b,c))
print(d) # [0 1 2 3 4 0 1 2 3 4]

################################# split #################################
a = np.arange(1, 10)
print(a)
'''
[1 2 3 4 5 6 7 8 9]
'''
b = np.split(a, 3)
print(b)
'''
[array([1, 2, 3]), array([4, 5, 6]), array([7, 8, 9])]
'''

a = np.arange(1, 13).reshape(6, 2)
print(a)
'''
[[ 1  2]
 [ 3  4]
 [ 5  6]
 [ 7  8]
 [ 9 10]
 [11 12]]
'''
b = np.split(a, 2)
b = np.vsplit(a, 2)
print(b)
'''
[array([[1, 2],
       [3, 4],
       [5, 6]]), 
array([[ 7,  8],
       [ 9, 10],
       [11, 12]])]
'''
b = np.split(a, 2, axis = 1)
b = np.hsplit(a, 2)
print(b)
'''
[array([[ 1],
       [ 3],
       [ 5],
       [ 7],
       [ 9],
       [11]]), 
array([[ 2],
       [ 4],
       [ 6],
       [ 8],
       [10],
       [12]])]
'''

################################# insert #################################
a = np.arange(1, 11)
# array, object
b = np.insert(a, 1, 50)
print(a) # [ 1  2  3  4  5  6  7  8  9 10]
print(b) # [ 1 50  2  3  4  5  6  7  8  9 10]

# insert float number into int64 array
a = np.arange(1, 11)
# array, object
b = np.insert(a, 1, 50.5) # truncated 
print(a) # [ 1  2  3  4  5  6  7  8  9 10]
print(b) # [ 1 50  2  3  4  5  6  7  8  9 10]

# insert 1 value at 2 position
a = np.arange(1, 11)
# array, object
b = np.insert(a, (1, 3), 50) # truncated 
print(a) # [ 1  2  3  4  5  6  7  8  9 10]
print(b) # [ 1 50  2  3 50  4  5  6  7  8  9 10]


# 2-d
a = np.array([[1, 2],[3, 4]])
print(a)
'''
[[1 2]
 [3 4]]
'''
b = np.insert(a, 1 , 23)
print(b) # [ 1 23  2  3  4] # ERROR: not like expect
# SOLUTION 1:*********************************** WEIRD
b = np.insert(a, 1 , 23, axis = 0)
print(b) 
'''
[[ 1  2]
 [23 23]
 [ 3  4]]
'''

# SOLUTION 2: *********************************** WEIRD
b = np.insert(a, 1 , 23, axis = 1)
print(b) 
'''
[[ 1 23  2]
 [ 3 23  4]]
'''

b = np.insert(a, 1, [10, 20], axis = 0)
print(b) 
'''
[[ 1  2]
 [10 20]
 [ 3  4]]
'''
# b = np.insert(a, 1, [10, 20, 30], axis = 0) # 
# ERROR: ValueError: could not broadcast input array from shape (1,3) into shape (1,2)



################################# append #################################
a = np.arange(1, 11)
b = np.append(a, 3452)
print(a) # [ 1  2  3  4  5  6  7  8  9 10]
print(b) # [   1    2    3    4    5    6    7    8    9   10 3452]

# 2d
a = np.array([[1, 2],[3, 4]])
b = np.append(a, [[4,5]], axis = 0)
print(b)
'''
[[1 2]
 [3 4]
 [4 5]]
'''

a = np.array([[1, 2],[3, 4]])
# b = np.append(a, [[4,5]], axis = 1) # ERROR: ValueError: all the input array dimensions except for the concatenation axis must match exactly
b = np.append(a, [[4],[5]], axis = 1)
print(b)
'''
[[1 2 4]
 [3 4 5]]
'''

# ERROR: 
b = np.append(a, [[4,5]])
print(b)
'''
[1 2 3 4 4 5]
'''

#help(np.append)

################################# append #################################
'''
delete index, not value
'''
a = np.arange(1, 11)
b = np.delete(a, 2)
print(a) # [ 1  2  3  4  5  6  7  8  9 10]
print(b) # [ 1  2  4  5  6  7  8  9 10]

# ERRROR: not expect
a = np.array([[1, 2],[3, 4]])
b = np.delete(a, 2)
print(b)
'''
[1 2 4]
'''

a = np.array([[1, 2],[3, 4]])
b = np.delete(a, 0, axis = 0)
print(b) 
'''
[[3 4]]
'''

a = np.array([[1, 2],[3, 4]])
b = np.delete(a, 0, axis = 1)
print(b) 
'''
[[2]
 [4]]
'''


