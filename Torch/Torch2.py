#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 09:55:36 2021

@author: tai
"""

import torch

####################### Tensor Indexing ######################
batch_size = 10
features = 25
x = torch.rand((batch_size, features))

print(x[0].shape)
print(x[0, :].shape)
'''
torch.Size([25])
'''

print(x[:,0].shape)
'''
torch.Size([10])
'''

print(x[2, 0:10])
'''
tensor([0.9286, 0.4898, 0.8987, 0.2071, 0.8699, 0.4331, 0.2616, 0.8916, 0.2866,
        0.5283])
'''
# Assign new value 
x[0, 0] = 100
print(x[0, 0])
'''
tensor(100.)
'''
# Fancy indexing
x = torch.arange(10)
indices = [2, 5, 8]
print(x)
print(x[indices])
'''
tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
tensor([2, 5, 8])
'''

x = torch.arange(15).reshape((3, 5))
rows = torch.tensor([1, 0])
cols = torch.tensor([4, 0])
print(x)
print(x[rows, cols])
print(x[rows, cols].shape)

'''
tensor([[ 0,  1,  2,  3,  4],
        [ 5,  6,  7,  8,  9],
        [10, 11, 12, 13, 14]])
tensor([9, 0])
torch.Size([2])
'''

# More advanced indexing
x = torch.arange(10)
print(x)
print(x[x % 2 == 0]) # Method 1:
print(x[x.remainder(2) == 0]) # Method 2:
print(x[(x % 2 == 0) & (x > 5)])
'''
tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
tensor([0, 2, 4, 6, 8])
tensor([6, 8])
'''


# Useful Operations
# Definition : where(condition, x, y) -> Tensor
print(torch.where(x > 5, x, x * 2))
'''
tensor([ 0,  2,  4,  6,  8, 10,  6,  7,  8,  9])
'''

print(torch.tensor([0, 0, 1, 2, 2, 3, 3, 4]).unique())
'''
tensor([0, 1, 2, 3, 4])
'''

# Check size and dimension of torch.tensor
print(x.ndimension()) # == numpy.shape :)) FUNCKING 
print(x.shape) # == numpy.size :)) FUNCKING
print(x.numel()) # == numpy.size :)) FUNCKING
'''
Ngược ngược với numpy :)) FUNCKING
1
torch.Size([10])
10
'''

###################################### Tensor Reshaping ######################
x = torch.arange(9)

y = x.view(3, 3)
print(y)
'''
tensor([[0, 1, 2],
        [3, 4, 5],
        [6, 7, 8]])
'''
y = x.reshape(3, 3)
print(y)
'''
tensor([[0, 1, 2],
        [3, 4, 5],
        [6, 7, 8]])
'''





# Transpose
x1 = torch.tensor([[1, 2, 3]])
print(x1)
print(x1.t())
'''
tensor([[1, 2, 3]])
tensor([[1],
        [2],
        [3]])
'''







# Concatenate 
x1 = torch.tensor([1, 2, 3])
x2 = torch.tensor([4, 5, 6])
y = torch.cat((x1, x2), dim = 0)
print(y)
'''
tensor([1, 2, 3, 4, 5, 6])
'''








# Flatten,
x = torch.tensor([[1, 2, 3],[4, 5, 6], [7, 8, 9]])

z = x.view(-1)
print(z)
'''
tensor([0, 1, 2, 3, 4, 5, 6, 7, 8])
'''

# Faltten each data point, NOT number of data/ batch_size
batch = 64
x = torch.rand((batch, 2, 5))
z = x.view(batch, -1)
print(z.shape)
'''
torch.Size([64, 10])
'''






# Change dimension order in torch.tensor
x = torch.rand((batch, 2, 5))
z = x.permute(0, 2, 1)
print(x.shape)
print(z.shape)
'''
torch.Size([64, 2, 5])
torch.Size([64, 5, 2])
'''






# Add dummy dimension before torch.tensor[0]
x = torch.arange(10)
print(x.unsqueeze(0).shape)
print(x.unsqueeze(0).unsqueeze(0).shape)
print(x.unsqueeze(1).unsqueeze(0).shape)
#print(x.unsqueeze(2).shape) # IndexError: Dimension out of range (expected to be in range of [-2, 1], but got 2)
print(x.unsqueeze(-1).shape) # add to the end
print(x.unsqueeze(-2).shape) # add to the front
'''
torch.Size([1, 10])
torch.Size([1, 1, 10])
torch.Size([1, 10, 1])
torch.Size([10, 1])
torch.Size([1, 10])
'''

x = torch.rand((batch, 2, 5))
'''
print(x.unsqueeze(4).shape)
IndexError: Dimension out of range (expected to be in range of [-4, 3], but got 4)
'''


# Remove 1 dimension of torch.tensor, Only when dimension is 1 Ex: 1x1x10
x = torch.arange(10).unsqueeze(0).unsqueeze(1) # 1x1x10
z = x.squeeze(0)
print(x.shape)
print(z.shape)
'''
torch.Size([1, 1, 10])
torch.Size([1, 10])
'''




x = torch.rand((batch, 2, 5))
z = x.squeeze(0)
print(x.shape)
print(z.shape)
'''
torch.Size([64, 2, 5])
torch.Size([64, 2, 5])
NOT work because 
'''















