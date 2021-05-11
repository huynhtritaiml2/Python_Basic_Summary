#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 17:05:07 2021

@author: tai
"""
'''
28px by 28px grayscale images of handwritten digits (0 to 9) 
and labels for each image indicating which digit it represents
'''
import torch
import torchvision
from torchvision.datasets import MNIST

dataset = MNIST(root='data/', download = True)
print(len(dataset)) # 60000

test_dataset = MNIST(root='data/', train = False)
print(len(test_dataset)) # 10000

print(dataset[0])
'''
(<PIL.Image.Image image mode=L size=28x28 at 0x7F4BABC2C3D0>, 5)
It's a pair, consisting of a 28x28px image and a label. 
The image is an object of the class PIL.Image.Image, 
which is a part of the Python imaging library Pillow. 
We can view the image within Jupyter using matplotlib, 
the de-facto plotting and graphing library for data science in Python.
'''

import matplotlib.pyplot as plt
#%matplotlib inline
'''
% are called magic commands and are used to configure the behavior of Jupyter itself. 
You can find a full list of magic commands here: https://ipython.readthedocs.io/en/stable/interactive/magics.html .
'''

image, label = dataset[0]
plt.imshow(image, cmap='gray')
print('Label: ', label) # Label:  5

def plot_sample(index):
    image, label = dataset[index]
    plt.imshow(image, cmap='gray')
    print('Label: ', label)
    
plot_sample(10) # Label:  3

'''
While it's useful to look at these images, there's just one problem here: 
PyTorch doesn't know how to work with images. We need to convert the images into tensors. 
We can do this by specifying a transform while creating our dataset.
'''

import torchvision.transforms as transforms
dataset = MNIST(root='data/', train=True, transform=transforms.ToTensor())

img_tensor, label = dataset[0]
print(img_tensor.shape, label) # torch.Size([1, 28, 28]) 5
'''
The image is now converted to a 1x28x28 tensor. 
The first dimension tracks color channels. 
The second and third dimensions represent pixels along the height and width of the image, respectively. 
Since images in the MNIST dataset are grayscale, there's just one channel. 
Other datasets have images with color, in which case there are three channels: red, green, and blue (RGB).
'''

print(img_tensor[0, 10:15, 10:15])
print(torch.max(img_tensor), torch.min(img_tensor))
'''
tensor([[0.0039, 0.6039, 0.9922, 0.3529, 0.0000],
        [0.0000, 0.5451, 0.9922, 0.7451, 0.0078],
        [0.0000, 0.0431, 0.7451, 0.9922, 0.2745],
        [0.0000, 0.0000, 0.1373, 0.9451, 0.8824],
        [0.0000, 0.0000, 0.0000, 0.3176, 0.9412]])
tensor(1.) tensor(0.)
'''

# Plot the image by passing in the 28x28 matrix
plt.imshow(img_tensor[0, 10:15, 10:15], cmap='gray')

from torch.utils.data import random_split
train_ds, val_ds = random_split(dataset, [50000, 10000])
print(len(train_ds)) # 50000
print(len(val_ds)) # 10000.

'''
It's essential to choose a random sample for creating a validation set. 
Training data is often sorted by the target labels, i.e., images of 0s, followed by 1s, followed by 2s, etc. 
If we create a validation set using the last 20% of images, it would only consist of 8s and 9s. 
In contrast, the training set would contain no 8s or 9s. Such a training-validation would make it impossible to train a useful model.

We can now create data loaders to help us load the data in batches. We'll use a batch size of 128.
'''

from torch.utils.data import DataLoader

batch_size = 128

train_loader = DataLoader(train_ds, batch_size, shuffle = True)
val_loader = DataLoader(val_ds, batch_size)

'''
We set shuffle=True for the training data loader to ensure that the batches 
generated in each epoch are different. This randomization helps generalize & speed up 
the training process. On the other hand, since the validation data loader 
is used only for evaluating the model, there is no need to shuffle the images.
'''
import torch.nn as nn
input_size = 28*28
num_classes = 10
class MnistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)
        
    def forward(self, xb):
        xb = xb.reshape(-1, 784)
        out = self.linear(xb)
        return out
    
model = MnistModel()

'''
Inside the __init__ constructor method, we instantiate the weights and biases using nn.Linear. 
And inside the forward method, which is invoked when we pass a batch of inputs to the model, 
we flatten the input tensor and pass it into self.linear.

xb.reshape(-1, 28*28) indicates to PyTorch that we want a view of the xb tensor with two dimensions. 
The length along the 2nd dimension is 28*28 (i.e., 784). One argument to .reshape can be set to -1 
(in this case, the first dimension) to let PyTorch figure it out automatically based on the shape of 
the original tensor.

Note that the model no longer has .weight and .bias attributes (as they are now inside 
the .linear attribute), but it does have a .parameters method that returns a list containing 
the weights and bias.
'''

print(model.linear)
'''
Linear(in_features=784, out_features=10, bias=True)
'''
print(model.linear.weight.shape, model.linear.bias.shape)
'''
torch.Size([10, 784]) torch.Size([10])
'''
print(list(model.parameters()))
'''
tensor([[-0.0231,  0.0342,  0.0013,  ..., -0.0316,  0.0191, -0.0297],
        [ 0.0356, -0.0211,  0.0037,  ..., -0.0047,  0.0143,  0.0341],
        [ 0.0238,  0.0130,  0.0058,  ..., -0.0242,  0.0338, -0.0307],
        ...,
        [ 0.0308, -0.0137,  0.0285,  ..., -0.0194,  0.0002, -0.0099],
        [ 0.0098, -0.0052, -0.0152,  ..., -0.0282,  0.0284,  0.0237],
        [ 0.0258,  0.0346, -0.0144,  ...,  0.0070,  0.0232, -0.0018]],
       requires_grad=True), Parameter containing:
tensor([-0.0228, -0.0289,  0.0303, -0.0341,  0.0306,  0.0256, -0.0237, -0.0228,
        -0.0056, -0.0268], requires_grad=True)]
'''

for images, labels in train_loader:
    print(images.shape) # torch.Size([128, 1, 28, 28])
    outputs = model(images)
    break

print(outputs.shape)
'''
torch.Size([128, 10])
'''
print(outputs[:2].data)
'''
tensor([[ 0.1692, -0.1067,  0.0192, -0.0500, -0.2906,  0.1285,  0.1673,  0.0284,
          0.0629,  0.1791],
        [-0.1405, -0.0009,  0.1330, -0.0350, -0.0385,  0.1992, -0.0055, -0.2011,
         -0.1070,  0.1129]])

For each of the 100 input images, we get 10 outputs, one for each class. 
As discussed earlier, we'd like these outputs to represent probabilities. 
Each output row's elements must lie between 0 to 1 and add up to 1, which is not the case.

To convert the output rows into probabilities, we use the softmax function, 
which has the following formula:

First, we replace each element yi in an output row by e^yi, making all the elements positive.

Then, we divide them by their sum to ensure that they add up to 1. 
The resulting vector can thus be interpreted as probabilities.

While it's easy to implement the softmax function (you should try it!), 
we'll use the implementation that's provided within PyTorch because 
it works well with multidimensional tensors (a list of output rows in our case).
'''


import torch.nn.functional as F

# Apply softmax for each output row
probs = F.softmax(outputs, dim = 1)


print(outputs[:2])
'''
tensor([[ 0.0048,  0.0422, -0.1218, -0.0761, -0.0506, -0.4828, -0.0366,  0.4266, 0.0889, -0.2732],
        [-0.1101,  0.1824, -0.1135, -0.1582,  0.0434,  0.1317,  0.1494,  0.3800, 0.3009, -0.1075]], grad_fn=<SliceBackward>)
'''
# Look at sample probabilities
print("Sample probabilities: \n", probs[:2].data)
'''
Sample probabilities: 
 tensor([[0.1028, 0.1067, 0.0905, 0.0948, 0.0972, 0.0631, 0.0986, 0.1567, 0.1118, 0.0778],
        [0.0822, 0.1101, 0.0819, 0.0783, 0.0958, 0.1047, 0.1065, 0.1342, 0.1240, 0.0824]])
'''
# Add up the probabilities of an output row
print("Sum: ", torch.sum(probs[0]))
'''
Sum:  tensor(1.0000, grad_fn=<SumBackward0>)
'''

'''
Finally, we can determine the predicted label for each image by simply choosing the index of the element 
with the highest probability in each output row. We can do this using torch.max, 
which returns each row's largest element and the corresponding index.


'''
'''

'''

max_probs, preds = torch.max(probs[:2], dim = 1)
print(preds)
'''
tensor([4, 1])
'''

print(max_probs)
'''
tensor([0.1311, 0.1304], grad_fn=<MaxBackward0>)
'''

print(labels[:2])
'''
tensor([3, 6])

Most of the predicted labels are different from the actual labels. 
That's because we have started with randomly initialized weights and biases. 
We need to train the model, i.e., adjust the weights using gradient descent to make better predictions.


'''

# Evaluation Metric and Loss Function

'''
Just as with linear regression, we need a way to evaluate how well our model is performing. 
A natural way to do this would be to find the percentage of labels that were predicted correctly, 
i.e,. the accuracy of the predictions.


'''
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim = 1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

'''
The == operator performs an element-wise comparison of two tensors with the same shape and returns a tensor of the same shape, containing True for unequal elements and False for equal elements. Passing the result to torch.sum returns the number of labels that were predicted correctly. Finally, we divide by the total number of images to get the accuracy.

Note that we don't need to apply softmax to the outputs since its results have the same relative order. This is because e^x is an increasing function, i.e., if y1 > y2, then e^y1 > e^y2. The same holds after averaging out the values to get the softmax.


'''

print(accuracy(outputs, labels))
'''
tensor(0.0312)
'''

'''
Accuracy is an excellent way for us (humans) to evaluate the model. 
However, we can't use it as a loss function for optimizing our model using gradient descent 
for the following reasons:

1. It's not a differentiable function. torch.max and == are both non-continuous and non-differentiable operations, 
so we can't use the accuracy for computing gradients w.r.t the weights and biases.

2. It doesn't take into account the actual probabilities predicted by the model, 
so it can't provide sufficient feedback for incremental improvements.

For these reasons, accuracy is often used as an evaluation metric for classification, 
but not as a loss function. A commonly used loss function for classification problems is the cross-entropy, 
which has the following formula:

While it looks complicated, it's actually quite simple:

- For each output row, pick the predicted probability for the correct label. 
E.g., if the predicted probabilities for an image are [0.1, 0.3, 0.2, ...] 
and the correct label is 1, we pick the corresponding element 0.3 and ignore the rest.

- Then, take the logarithm of the picked probability. If the probability is high, i.e., close to 1, 
then its logarithm is a very small negative value, close to 0. And if the probability is low (close to 0), 
then the logarithm is a very large negative value. We also multiply the result by -1, 
which results is a large postive value of the loss for poor predictions.    

- Finally, take the average of the cross entropy across all the output rows 
to get the overall loss for a batch of data.

Unlike accuracy, cross-entropy is a continuous and differentiable function. 
It also provides useful feedback for incremental improvements in the model 
(a slightly higher probability for the correct label leads to a lower loss). 
These two factors make cross-entropy a better choice for the loss function.

As you might expect, PyTorch provides an efficient and tensor-friendly implementation of cross-entropy 
as part of the torch.nn.functional package. Moreover, it also performs softmax internally, 
so we can directly pass in the model's outputs without converting them into probabilities.
'''

loss_fn = F.cross_entropy
loss = loss_fn(outputs, labels)
print(loss)
'''
tensor(2.3201, grad_fn=<NllLossBackward>)
We know that cross-entropy is the negative logarithm of the predicted probability of 
the correct label averaged over all training samples. Therefore, one way to interpret 
the resulting number e.g. 2.23 is look at e^-2.23 which is around 0.1 as the predicted probability 
of the correct label, on average. The lower the loss, The better the model.
'''

# Training the model

'''
for epoch in range(num_epochs):
    # Training phase
    for batch in train_loader:
        # Generate predictions
        # Calculate loss
        # Compute gradients
        # Update weights
        # Reset gradients
    
    # Validation phase
    for batch in val_loader:
        # Generate predictions
        # Calculate loss
        # Calculate metrics (accuracy etc.)
    # Calculate average validation loss & metrics
    
    # Log epoch, loss & metrics for inspection
'''

def evaluate(model, val_loader):
    outputs = [model.validation(batch) for batch in val_loader]
    return model.validation_mean(outputs)


def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    optimizer = opt_func(model.parameters(), lr)
    history = [] # Recording Epoch Result
    for epoch in range(epochs):
        # Training Phase
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result)
        history.append(result)
        
    return history
'''
The fit function records the validation loss and metric from each epoch. 
It returns a history of the training, useful for debugging & visualization.

Configurations like batch size, learning rate, etc. (called hyperparameters), 
need to picked in advance while training machine learning models. 
Choosing the right hyperparameters is critical for training a reasonably accurate model 
within a reasonable amount of time. It is an active area of research and experimentation 
in machine learning. Feel free to try different learning rates and see how 
it affects the training process. 
'''       
        
class MnistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)
        
    def forward(self, xb):
        xb = xb.reshape(-1, 784)
        out = self.linear(xb)
        
        return out
    
    def training_step(self, batch): # training is in nn.Module, change the name
        images, labels = batch
        out = self(images)  # Generate Predicions
        loss = F.cross_entropy(out, labels) # Calculate Loss
        
        return loss
    
    def validation(self, batch):
        images, labels = batch
        out = self(images) # Generate Predicions
        loss = F.cross_entropy(out, labels) # Calculate Loss
        acc = accuracy(out, labels) # Calculate accuracy
        
        return {'val_loss': loss, 'val_acc': acc}

    def validation_mean(self, outputs):
        batch_losses = [dic['val_loss']  for dic in outputs]
        epoch_loss = torch.stack(batch_losses).mean() # # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean() # Combine accuracies
        
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))
        
model = MnistModel()
result0 = evaluate(model, val_loader)
print(result0)
'''
{'val_loss': 2.3129873275756836, 
 'val_acc': 0.09651898592710495}


The initial accuracy is around 10%, which one might expect from a randomly initialized model (since it has a 1 in 10 chance of getting a label right by guessing randomly).

We are now ready to train the model. Let's train for five epochs and look at the results.
'''
history1 = fit(5, 0.001, model, train_loader, val_loader)
history2 = fit(5, 0.001, model, train_loader, val_loader)
history3 = fit(5, 0.001, model, train_loader, val_loader)
history4 = fit(5, 0.001, model, train_loader, val_loader)
'''
Epoch [0], val_loss: 1.9542, val_acc: 0.5937
Epoch [1], val_loss: 1.6845, val_acc: 0.7218
Epoch [2], val_loss: 1.4825, val_acc: 0.7658
Epoch [3], val_loss: 1.3293, val_acc: 0.7851
Epoch [4], val_loss: 1.2116, val_acc: 0.7981

Epoch [0], val_loss: 1.1190, val_acc: 0.8083
Epoch [1], val_loss: 1.0447, val_acc: 0.8160
Epoch [2], val_loss: 0.9839, val_acc: 0.8218
Epoch [3], val_loss: 0.9334, val_acc: 0.8268
Epoch [4], val_loss: 0.8906, val_acc: 0.8314

Epoch [0], val_loss: 0.8540, val_acc: 0.8350
Epoch [1], val_loss: 0.8223, val_acc: 0.8385
Epoch [2], val_loss: 0.7945, val_acc: 0.8409
Epoch [3], val_loss: 0.7700, val_acc: 0.8427
Epoch [4], val_loss: 0.7482, val_acc: 0.8441

Epoch [0], val_loss: 0.7286, val_acc: 0.8457
Epoch [1], val_loss: 0.7109, val_acc: 0.8474
Epoch [2], val_loss: 0.6949, val_acc: 0.8490
Epoch [3], val_loss: 0.6803, val_acc: 0.8511
Epoch [4], val_loss: 0.6669, val_acc: 0.8527
'''
plt.show()
history = [result0] + history1 + history2 + history3 + history4
acc = [dic['val_acc'] for dic in history]
plt.plot(acc, '-x')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Accuracy vs #of epochs')
plt.show()

'''
It's quite clear from the above picture that the model probably won't cross the accuracy threshold of 90%
 even after training for a very long time. One possible reason for this is that the learning rate 
 might be too high. The model's parameters may be "bouncing" around the optimal set of parameters 
 for the lowest loss. You can try reducing the learning rate and training for a few more epochs to see 
 if it helps.

The more likely reason that the model just isn't powerful enough. If you remember our initial hypothesis, 
we have assumed that the output (in this case the class probabilities) is a linear function of the input 
(pixel intensities), obtained by perfoming a matrix multiplication with the weights matrix and 
adding the bias. This is a fairly weak assumption, as there may not actually exist a linear relationship 
between the pixel intensities in an image and the digit it represents. While it works reasonably well 
for a simple dataset like MNIST (getting us to 85% accuracy), we need more sophisticated models 
that can capture non-linear relationships between image pixels and labels 
for complex tasks like recognizing everyday objects, animals etc.
'''

test_dataset = MNIST(root='data/', train=False, transform=transforms.ToTensor())

def predict_img(img_index):
    img, label = test_dataset[img_index]
    plt.imshow(img[0], cmap='gray') # img[0], because only plot 1 dimension
    plt.show()
    
    img_tensor = img.unsqueeze(0)
    predict = model(img_tensor)
    _, pred = torch.max(predict, dim=1)
    print('Label:', label, ', Predicted: ', pred)

'''
img.unsqueeze simply adds another dimension at the begining of the 1x28x28 tensor, making it a 1x1x28x28 tensor, which the model views as a batch containing a single image.


'''
print("Predict Image")
predict_img(0)
predict_img(10)
predict_img(193)
predict_img(1839)
'''
Label: 7 , Predicted:  tensor([7])
Label: 0 , Predicted:  tensor([0])
Label: 9 , Predicted:  tensor([4])
Label: 2 , Predicted:  tensor([8])
'''

test_loader = DataLoader(test_dataset, batch_size=256)
result = evaluate(model, test_loader)
print(result)
'''
{'val_loss': 0.6406440138816833, 'val_acc': 0.85986328125}
We expect this to be similar to the accuracy/loss on the validation set. 
If not, we might need a better validation set that has similar data and distribution 
as the test set (which often comes from real world data).
'''
# Saving and loading the model
'''
Since we've trained our model for a long time and achieved a resonable accuracy, 
it would be a good idea to save the weights and bias matrices to disk, 
so that we can reuse the model later and avoid retraining from scratch. 
Here's how you can save the model.



'''
print("Save Model")
torch.save(model.state_dict(), 'MNIST-logistic.pth')
'''
The .state_dict method returns an OrderedDict containing all the weights and bias matrices mapped to the right attributes of the model.

'''
print(model.state_dict())
'''
OrderedDict([('linear.weight', tensor([[-0.0292, -0.0335, -0.0131,  ..., -0.0227,  0.0237, -0.0245],
        [-0.0002,  0.0230,  0.0222,  ..., -0.0053,  0.0160, -0.0298],
        [ 0.0105,  0.0346,  0.0200,  ..., -0.0100, -0.0099, -0.0155],
        ...,
        [ 0.0249, -0.0123, -0.0032,  ..., -0.0301,  0.0145, -0.0264],
        [ 0.0246, -0.0090,  0.0325,  ...,  0.0186, -0.0243, -0.0137],
        [ 0.0171, -0.0191, -0.0024,  ...,  0.0149,  0.0305, -0.0278]])), ('linear.bias', tensor([-0.0534,  0.0874, -0.0492, -0.0431,  0.0431,  0.0728, -0.0288,  0.0203,
        -0.0612,  0.0017]))])
                                                                                                 '''

'''
To load the model weights, we can instante a new object of the class MnistModel, and use the .load_state_dict method.
'''
model2 = MnistModel()
print(model2.state_dict())
print(evaluate(model2, test_loader))
''''
{'val_loss': 2.357172727584839, 'val_acc': 0.08632812649011612}

'''
model2.load_state_dict(torch.load('MNIST-logistic.pth'))
print(model2.state_dict())
print(evaluate(model2, test_loader))
'''
{'val_loss': 0.6406440138816833, 'val_acc': 0.85986328125}
'''
test_loader = DataLoader(test_dataset, batch_size=256)
result = evaluate(model2, test_loader)
print(result)
'''
{'val_loss': 0.6406440138816833, 'val_acc': 0.85986328125}
'''















    


