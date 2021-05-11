#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 21:32:06 2021

@author: tai
"""

# Imports
import torch
import torch.nn as nn # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss Function
import torch.optim as optim # adam, sgd
import torch.nn.functional as F # relu, tanh, sigmoid, All function that don't have any parameter
from torch.utils.data import DataLoader # Give easier dataset managment and create mini batches
import torchvision.datasets as datasets # Has standard datasets we can import in a nice way 
import torchvision.transforms as transforms # Transformation we can perform on the dataset

# 1. Create Fully COnnect Network
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__() # In this case == super().__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# TODO: Create simple CNN
class CNN(nn.Module):
    def __init__(self, in_channel = 1, num_classes = 10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 8, kernel_size=(3,3), stride=(1,1), padding=(1,1)) # Output have the same size of original img
        '''
        n_out = (n_in + 2*p - k)/ s +1
        p: padding
        s: stride
        k: kernel size
        '''
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)) # n_out = (28 + 2*0 - 2)/ 2 + 1 = 7
        self.conv2 = nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = (3,3), stride = (1,1), padding = (1, 1))
        self.fc1 = nn.Linear(16*7*7, num_classes)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1) # Flatten the output
        x = self.fc1(x)
        
        return x

'''        
## Test
model = CNN()
x = torch.rand(64, 1, 28, 28) # torch.Size([64, 1, 28, 28])
print(x.shape)
print(model(x).shape) # torch.Size([64, 10])
exit()
'''

# 2. Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3. Hyperparameter
#input_size = 784 # 28 * 28
in_channel = 1

num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1


# 4. Load Data
# because when it load data, it maybe in form of numpy
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 5. Initialize network
#model = NN(input_size = input_size, num_classes=num_classes).to(device)
model = CNN().to(device)

# 6. Loss and optimizer
criterion = nn.CrossEntropyLoss() # Sparse, Categorical Cross Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# 7. Train Network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to CUDA if possible
        data = data.to(device=device)
        targets = targets.to(device=device)
        
        #print(data.shape)
        # Get to correct shape
        #data = data.reshape(data.shape[0], -1) # We have do it in CNN()*********************
        
        # Forward
        scores = model(data) # pass output of each layer x -> z -> a
        loss = criterion(scores, targets) # Compare only output of the last layer and y_labels
        
        # Backward
        optimizer.zero_grad() # Clear previous gradient value, or previous backward of previous batch
        loss.backward() # caluclate delta W baseod on loss function, and backward to previous layers
        
        # Gradient Descent or Adam step
        optimizer.step() # w = w - learning_rate * delta_w
        
# 8. Check accuracy on training and test to see how good our model

def check_accuracy(loader, model):
    if(loader.dataset.train):
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on testing data")
    
    num_correct = 0
    num_samples = 0
    model.eval()
    
    with torch.no_grad(): # no need to calculate the gradient
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            #x = x.reshape(x.shape[0], -1)  # We have do it in CNN() *********************
            
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        
        print(f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}")
        
    model.train()
    #return acc

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)

'''
Fully Connected Network:
Output from spyder call 'get_namespace_view':
Checking accuracy on training data
Got 56000 / 60000 with accuracy 93.33
Checking accuracy on testing data
Got 9348 / 10000 with accuracy 93.48    
    
    
CNN:
Checking accuracy on training data
Got 57884 / 60000 with accuracy 96.47
Checking accuracy on testing data
Got 9666 / 10000 with accuracy 96.66    

'''