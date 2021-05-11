#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 07:10:21 2021

@author: tai
https://www.youtube.com/watch?v=x9JiIFvlUwk&list=PLhhyoLH6IjfxeoooqP9rhU3HJIAVAJ3Vz&index=2
"""

import torch

print(torch.__version__) # 1.5.0

#x = torch.rand(64, 10)
#

my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
'''
tensor([[1, 2, 3],
        [4, 5, 6]])
'''
my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
'''
tensor([[1., 2., 3.],
        [4., 5., 6.]])
'''
my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device="cuda")
'''
tensor([[1., 2., 3.],
        [4., 5., 6.]], device='cuda:0')
We have CUDA GPU
'''
my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device="cpu")
'''
tensor([[1., 2., 3.],b
        [4., 5., 6.]])
CPU is default
'''
my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device="cuda", requires_grad=True)
'''
tensor([[1., 2., 3.],
        [4., 5., 6.]], device='cuda:0', requires_grad=True)
For compute Gradient Descent:
    
'''
# Method 2: for activate cude if we have
device = "cuda" if torch.cuda.is_available() else "cpu"
my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device=device, requires_grad=True)
'''
tensor([[1., 2., 3.],
        [4., 5., 6.]], device='cuda:0', requires_grad=True)
'''

# Attribute
print(my_tensor) # print all array/tensor like above
print(my_tensor.dtype) # torch.float32
print(my_tensor.device) # cuda:0
print(my_tensor.shape) # torch.Size([2, 3])
print(my_tensor.requires_grad) # True


# Other comman Initialization method
x = torch.empty(size = (3, 3))
'''
tensor([[1.0253e+08, 3.0928e-41, 8.4078e-45],
        [0.0000e+00, 5.0000e+00, 6.0000e+00],
        [4.3722e-05, 2.1670e-04, 2.6192e+20]])
This value is current value at that space, not is initialized
'''

x = torch.zeros((3,3))
'''
tensor([[0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]])
'''

x = torch.rand((3, 3))
'''
tensor([[0.1816, 0.1473, 0.1198],
        [0.4418, 0.2482, 0.3609],
        [0.4274, 0.2651, 0.5129]])
'''

x = torch.ones((3 , 3))
'''
tensor([[1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.]])
'''

x = torch.eye(5)
'''
tensor([[1., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0.],
        [0., 0., 1., 0., 0.],
        [0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 1.]])
'''

x = torch.eye(3, 3)
'''
NOT: ERROR: x = torch.eye((3, 3))
tensor([[1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.]])
'''

x = torch.arange(start=1, end=6, step=2) # tensor([1, 3, 5])
x = torch.linspace(start=0.1, end=1, steps=10)

'''
Create 10 value from a certain range
tensor([0.1000, 0.2000, 0.3000, 0.4000, 0.5000, 0.6000, 0.7000, 0.8000, 0.9000,
        1.0000])
'''

x = torch.empty(size=(1, 5)).normal_(mean=0, std=1)
'''
tensor([[-0.1700,  0.8621, -0.2922, -0.8947,  0.8784]])
'''
x = torch.empty(size=(1, 5)).uniform_(0, 1)
'''
tensor([[0.8757, 0.2174, 0.6552, 0.6180, 0.6043]])
'''

x = torch.diag(torch.ones(3))
'''
tensor([[1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.]])
'''



# How to initialize and convert tensors to other types (int, float, double)
tensor = torch.arange(4)
print(tensor.bool())
'''
tensor([False,  True,  True,  True])
'''
print(tensor.short())
'''
tensor([0, 1, 2, 3], dtype=torch.int16)
'''
print(tensor.long())
'''
tensor([0, 1, 2, 3])
'''
print(tensor.half())
'''
tensor([0., 1., 2., 3.], dtype=torch.float16)
# NOTE: GPU must support float16
'''
print(tensor.float())
'''
tensor([0., 1., 2., 3.])
'''
print(tensor.double())
'''
tensor([0., 1., 2., 3.], dtype=torch.float64)
'''




# Array to Tensor conversion and vice-versa
import numpy as np

np_array = np.zeros((2, 4))
print(np_array)
'''
[[0. 0. 0. 0.]
 [0. 0. 0. 0.]]
'''

tensor = torch.from_numpy(np_array) # Method 1:
tensor = torch.tensor(np_array) # Method 2:
print(tensor)
'''
tensor([[0., 0., 0., 0.],
        [0., 0., 0., 0.]], dtype=torch.float64)
'''

np_array_back = tensor.numpy() # Method 1:
np_array_back = np.array(tensor) # Method 2:
print(np_array_back)
'''
[[0. 0. 0. 0.]
 [0. 0. 0. 0.]]
'''

# Tensor Math & Comparison Operation

x = torch.tensor([1, 2, 3])
y = torch.tensor([9, 8, 7])


# Addition
# Method 1:
z1 = torch.empty(3)
torch.add(x, y, out=z1)
print(z1) 
'''
tensor([10., 10., 10.])
'''
# Method 2:
z2 = torch.add(x, y)
print(z2)
'''
tensor([10, 10, 10])
'''
# Method 3:
z = x + y
print(z)
'''
tensor([10, 10, 10])
'''

# Substraction
z = x - y
print(z)
'''
tensor([-8, -6, -4])
'''

# Division
z = x / y
print(z)
'''
tensor([0, 0, 0])
/pytorch/aten/src/ATen/native/BinaryOps.cpp:81: 
UserWarning: Integer division of tensors using div or / is deprecated, 
and in a future release div will perform true division as in Python 3. 
Use true_divide or floor_divide (// in Python) instead.
'''
z = torch.true_divide(y, x)
print(z)
'''
x / y
tensor([0.1111, 0.2500, 0.4286])
y / x
tensor([9.0000, 4.0000, 2.3333])
'''

z = torch.floor_divide(y, x)
print(z)
'''
x / y
tensor([0, 0, 0])
y / x
tensor([9, 4, 2])
'''

z = y // x # Similar to torch.floor_divide(y, x)
print(z)
'''
tensor([9, 4, 2])
'''

# Exponentiation
z = x.pow(2) # Method 1:
print(z)
'''
tensor([1, 4, 9])
'''

z = x ** 2 # Method 2:
print(z)
'''
tensor([1, 4, 9])
'''


# Modular
z = y % x
print(z)
'''
tensor([0, 0, 1])
'''



# Inplace Operations
t = torch.zeros(3)
t.add_(x)
t += x # t = t + x
print(t) # tensor([2., 4., 6.])



# Simpe Comparasion
z = x > 0
print(z)
'''
tensor([True, True, True])
'''
z = x < 0
print(z)
'''
tensor([False, False, False])
'''
# Matrix Multiplication == np.dot(x,y)
print("Matrix Multiplcation ###########################")
x1 = torch.arange(6).reshape(2, 3)
x2 = torch.arange(6).reshape(3, 2)
x3 = torch.mm(x1, x2) # Method 1:
print(x1)
print(x2)
print(x3)
'''
# 2x3 matrix
tensor([[0, 1, 2],
        [3, 4, 5]])
tensor([[0, 1],
        [2, 3],
        [4, 5]])
tensor([[10, 13],
        [28, 40]])
'''

x3 = x1.mm(x2) # Method 2:
print(x3)
'''
tensor([[10, 13],
        [28, 40]])
'''

# Matrix Exponentiation == np.pow(x, 2)
#matrix_exp = torch.tensor([1, 2, 3]) # RuntimeError: matrix_power(Long{[3]}): expected a tensor of floating types with dim at least 2
#matrix_exp = torch.tensor([[1, 2, 3],[7, 8, 9]], dtype=torch.float32) # size mismatch, m1: [2 x 3], m2: [2 x 3] at /pytorch/aten/src/TH/generic/THTensorMath.cpp:41
matrix = torch.tensor([[1, 2],[7, 8]], dtype=torch.float32)
matrix_exp = matrix.matrix_power(2)
print(matrix_exp)
'''
matrix_exp = matrix.matrix_power(2)
tensor([[15., 18.],
        [63., 78.]])

matrix_exp = matrix.matrix_power(3)
tensor([[141., 174.],
        [609., 750.]])
'''


# Elelemt Wise Mult : == x * y in numpy
x = torch.tensor([1, 2, 3])
y = torch.tensor([9, 8, 7])
z = x * y
print(z)
'''
tensor([ 9, 16, 21])
'''

# Dot Product, ONLY for 1D FUCKING :)) complexer than numpy.dot
'''
x = torch.arange(6).reshape(2, 3)
y = torch.arange(6).reshape(3, 2)
ERROR
# 1D tensors expected, got 2D, 2D tensors at /pytorch/aten/src/TH/generic/THTensorEvenMoreMath.cpp:431
'''
z = torch.dot(x, y)

print(z) 
'''
tensor(46) = 1*9 + 2*8 + 3*7 = 46
'''

# Bath Matrix Multiplication, ONLY for 3D :)) FUCKING complexer than numpy.dot
batch = 32
n = 10
m = 20
p = 38

tensor1 = torch.rand((batch, n, m))
tensor2 = torch.rand((batch, m, p))
out_bmm = torch.bmm(tensor1, tensor2) 
print(out_bmm.shape)
'''
matrix(batch, n, m) * matrix(batch, m, p) = 
torch.Size([32, 10, 38])
-> (batch, n, p)

'''

# Example of Broadcasting
x1 = torch.tensor([[1, 2, 3], [3, 4, 5]])
x2 = torch.tensor([10, 11, 12])

z = x1 + x2
print(z)
'''
tensor([[11, 13, 15],
        [13, 15, 17]])
'''

z = x1 ** x2
print(z)
'''
tensor([[        1,      2048,    531441],
        [    59049,   4194304, 244140625]])
'''

# Other userful tensor operations
x = torch.tensor([[1, 2, 3], [3, 4, 5]]) # Method 1:
sum_x = torch.sum(x, dim=0) # Method 2:
print(sum_x)
'''
tensor(6)
'''

values, indexs = torch.max(x, dim = 0) # Method 1:
values, indexs = x.max(dim = 0) # Method 2:
print(f"values : {values}, indexs = {indexs}")
'''
values : tensor([3, 4, 5]), indexs = tensor([1, 1, 1])
'''
values, indexs = torch.min(x, dim = 0) # Method 1:
values, indexs = x.min(dim = 0) # Method 2:
print(f"values : {values}, indexs = {indexs}")
'''
values : tensor([1, 2, 3]), indexs = tensor([0, 0, 0])
'''

x = torch.tensor([[-1, -2, -3], [3, 4, 5]])
abs_x = torch.abs(x)
print(abs_x)
'''
tensor([[1, 2, 3],
        [3, 4, 5]])
'''


z = torch.argmax(x, dim=0)
print(z)
'''
tensor([1, 1, 1])
'''


mean_x = torch.mean(x.float(), dim=0)
print(mean_x)
'''
tensor([1., 1., 1.])
'''

# Equal, compare between 2 matrix
x = torch.tensor([1, 2, 3])
y = torch.tensor([3, 2, 1])
z = torch.eq(x, y)
print(z)
'''
tensor([False,  True, False])
'''


# Sort the torch tensor
y = torch.tensor([1, 2, 4, 10, 8, 7])
y_sorted, indices = torch.sort(y, dim=0, descending=False)
print(f"y sorted: {y_sorted}, indices: {indices}")
'''
y sorted: 
tensor([ 1,  2,  4,  7,  8, 10]), 
indices: 
tensor([0, 1, 2, 5, 4, 3]) 
indices mean, swap index,
'''


# Fix the certain range value for torch.tensor
x = torch.tensor([1, 2, 3, 5, 6, 9, 8, 7, 5, 10])
z = torch.clamp(x, min = 2, max = 5)
print(z)
'''
tensor([2, 2, 3, 5, 5, 5, 5, 5, 5, 5])
'''


# OR operation, AND operation for bool type torch.tensor
#z = torch.any(x) # RuntimeError: all only supports torch.uint8 and torch.bool dtypes
x = torch.tensor([0, 1, 1, 1, 0], dtype=torch.bool)
z = torch.any(x)
print(z)
'''
tensor(True)
'''


z = torch.all(x)
print(z)
'''
tensor(False)
'''





























