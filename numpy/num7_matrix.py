import numpy as np
'''
matrix multiplication : dot()
'''
a = np.array([[10,20],[30,40]])
b = np.array([[1,2],[3,4]])

c = a * b
print(c) 
'''
Array Multiplication NOT Matrix multiplication
[[ 10  40]
 [ 90 160]]
'''

# matrix multiplication
c = a.dot(b)
print(c)
'''
[[ 70 100]
 [150 220]]
'''

########################## Transpose
a = np.array([[1, 2, 3],[4,5,6]])
print(a.T)
print(np.transpose(a))
'''
[[1 4]
 [2 5]
 [3 6]]
[[1 4]
 [2 5]
 [3 6]]
'''

########################### CREATE a MATRIX #################################
# METHOD 1: string
a = np.matrix("1 2;3 4")
print(a)
'''
[[1 2]
 [3 4]]
'''

# METHOD 2: list
a = np.matrix([[10, 20],[30, 40]])
print(a)
'''
[[10 20]
 [30 40]]
'''

########################### MATRIX Operation #################################
a = np.matrix("1 2;3 4")
b = np.matrix([[10, 20],[30, 40]])

print(a + b)
'''
similar to array
[[11 22]
 [33 44]]
'''
# Matrix multiplication 
print(a*b) # similar to a.dot(b) **********************************
'''
[[ 70 100]
 [150 220]]
'''

# Transpose
print(a.T)
'''
[[1 3]
 [2 4]]
'''
########################### MATRIX Operation #################################
'''
no.linalg
1. invert (A-1 or A.dot(A-1) = I ) 
no.linalg.inv(a) :only square matrix can be inversed using inv(): Ma trận vuông mới inverse được 
https://mrwilliamsmath.weebly.com/pre-ap-algebra-ii/inverse-matrices
2. power of matrix
3. linear equation
4. determinant
'''
################################# Invert matrix #################################
a = np.array([[1,2],[3,4]])
print(a)
b = np.linalg.inv(a)
print(b)
'''
[[1 2]
 [3 4]]
[[-2.   1. ]
 [ 1.5 -0.5]]
'''

print(a.dot(b))
'''
I Matrix: unit matrix
[[1.0000000e+00 0.0000000e+00]
 [8.8817842e-16 1.0000000e+00]]
'''
################################# matrix Power #################################
'''
np.linalg.matrix_power(a,n) : WORK for Square matrix *************************
n = 0 -> Identity Matrix
n > 0 -> 
n < 0 -> Inverse()
'''
a = np.array([[1,2],[3,4]])
print(a)
n = 2
b = np.linalg.matrix_power(a, n)
print(b)
'''
A^2
[[1 2]
 [3 4]]
[[ 7 10]
 [15 22]]
'''

# Similar to 
print(a.dot(a))
'''
[[ 7 10]
 [15 22]]
'''

a = np.array([[1,2],[3,4]])
print(a)
n = 0
b = np.linalg.matrix_power(a, n)
print(b)
'''
A^0
[[1 0]
 [0 1]]
'''

a = np.array([[1,2],[3,4]])
print(a)
n = -2
b = np.linalg.matrix_power(a, n)
print(b)
'''
A^-2
[[ 5.5  -2.5 ]
 [-3.75  1.75]]
'''

# Similar to A^-1 and take power
a = np.array([[1,2],[3,4]])
b = np.linalg.inv(a)
b = np.linalg.matrix_power(b, 2)
print(b)
'''
[[ 5.5  -2.5 ]
 [-3.75  1.75]]
'''

#ERROR
a = np.array([[1,2,3],[3,4,5]])
#b = np.linalg.matrix_power(a, 2) # numpy.linalg.LinAlgError: Last 2 dimensions of the array must be square

################################# Solving Linear Equation #################################
'''
3x + 1 = 9
1x + 2 = 8

Ax = B
'''
a = np.array([[3, 1],[1, 2]])
b = np.array([9,8])

x = np.linalg.solve(a, b) # [2. 3.]
print(x) 


'''
6x + 2y -5z = 13
3x + 3y -2z = 13
7x + 5y -3z = 26

Ax = B
'''
a = np.array([[6, 2, -5],[3, 3, -2], [7, 5, -3]])
b = np.array([13,13,26])
x = np.linalg.solve(a, b) # [2. 3. 1.]
print(x) 

################################# Determinant #################################
'''
a b = ad - bc
c d
'''

a = np.array([[1,2],[3,4]])
b = np.linalg.det(a)
print(b) # -2.0000000000000004
print(round(b)) # -2.0


a = np.array([[1,2,3],[4,5,6],[7,8,9]])
b = np.linalg.det(a)
print(b) # 0.0
 
