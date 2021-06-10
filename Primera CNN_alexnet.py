#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 12:33:30 2021

@author: nataly

Your first CNN - __init__ method
You are going to build your first convolutional neural network. You're going 
to use the MNIST dataset as the dataset, which is made of handwritten digits 
from 0 to 9. The convolutional neural network is going to have 2 convolutional
 layers, each followed by a ReLU nonlinearity, and a fully connected layer. 
 We have already imported torch and torch.nn as nn. Remember that each pooling
 layer halves both the height and the width of the image, so by using 2 pooling 
 layers, the height and width are 1/4 of the original sizes. MNIST images have 
 shape (1, 28, 28)

For the moment, you are going to implement the __init__ method of the net. In
 the next exercise, you will implement the .forward() method.

NB: We need 2 pooling layers, but we only need to instantiate a pooling layer 
once, because each pooling layer will have the same configuration. Instead, 
we will use self.pool twice in the next exercise.
"""

import torch
import torchvision
import torch.utils.data
import torchvision.transforms as transforms

import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Transform the data to torch tensors and normalize it 
transform = transforms.Compose([transforms.ToTensor(),
								transforms.Normalize
								((0.1307),(0.3081))])

# Prepare training set and testing set
trainset = torchvision.datasets.MNIST('mnist', train=True, 
									  download=True, transform=transform)

testset = torchvision.datasets.MNIST('mnist', train=False, download=True, transform=transform)

# Prepare training loader and testing loader
trainloader = torch.utils.data.DataLoader(trainset,batch_size=32,
                                          shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=32,
										 shuffle=False, num_workers=0)


class Net(nn.Module):
    def __init__(self, num_classes=10):
        super(Net, self).__init__()
		
        # Instantiate the ReLU nonlinearity
        self.relu = nn.ReLU()
        
        # Instantiate two convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=5, out_channels=10, kernel_size=3, padding=1)
        
        # Instantiate a max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Instantiate a fully connected layer
        self.fc = nn.Linear(7 * 7 * 10, 10)

    def forward(self, x):

        # Apply conv followd by relu, then in next line pool
        x = self.relu(self.conv1(x))
        x = self.pool(x)

        # Apply conv followd by relu, then in next line pool
        x = self.relu(self.conv2(x))
        x = self.pool(x)

        # Prepare the image for the fully connected layer
        x = x.view(-1, 7*7*10)

        # Apply the fully connected layer and return the result
        return self.fc(x)


"""
Training CNNs
Similarly to what you did in Chapter 2, you are going to train a neural network.
 This time however, you will train the CNN you built in the previous lesson, 
 instead of a fully connected network. The packages you need have been 
 imported for you and the network (called net) instantiated. 
 The cross-entropy loss function (called criterion) and the Adam optimizer 
 (called optimizer) are also available. We have subsampled the training set
 so that the training goes faster, and you are going to use a single epoch.
 
"""
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=3e-4)

for epoch in range(10):  
    
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
    
        # Compute the forward pass
        outputs = net(inputs)
            
        # Compute the loss function
        loss = criterion(outputs,labels)
            
        # Compute the gradients
        loss.backward()
        
        # Update the weights
        optimizer.step()
        print('Finished Training')

correct, total = 0, 0
predictions = []
net.eval()
# Iterate over the data in the test_loader
for i, data in enumerate(testloader,0):

    # Get the image and label from data
    inputs, labels = data

    # Make a forward pass in the net with your image
    outputs = net(inputs)
    _, predicted = torch.max(outputs.data, 1)
    predictions.append(outputs)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
    print('The testing set accuracy of the network is: %d %%' % (
            100 * correct / total))
    # Argmax the results of the net
    #_, predicted = torch.max(output.data, 1)
    #if predicted == labels:
     #  print("Yipes, your net made the right prediction " + str(predicted))
    #else:
     #  print("Your net prediction was " + str(predicted) + ", but the correct label is: " + str(labels))
        
        
        
        
        