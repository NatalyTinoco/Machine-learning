#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 11:14:15 2021

@author: nataly

Validation set
You saw the need for validation set in the previous video. Problem is that 
the datasets typically are not separated into training, validation and testing. 
It is your job as a data scientist to split the dataset into training, testing 
and validation. The easiest (and most used) way of doing so is to do a random 
splitting of the dataset. In PyTorch, that can be done using SubsetRandomSampler
 object. You are going to split the training part of MNIST dataset into training
 and validation. After randomly shuffling the dataset, use the first 55000 points 
 for training, and the remaining 5000 points for validation.
"""
import torch
import torchvision
import torch.utils.data
import torchvision.transforms as transforms
import torch.optim as optim

transform = transforms.Compose([transforms.ToTensor(),
transforms.Normalize((0.4914, 0.48216, 0.44653),
(0.24703, 0.24349, 0.26159))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(testset, batch_size=32,shuffle=False, num_workers=4)


"""
Recap-Dataloaders
Preparing MNIST dataset
"""

import torch
import torchvision
import torch.utils.data

import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn

# Shuffle the indices
indices = np.arange(60000)
np.random.shuffle(indices)

# Build the train loader
train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('mnist', download=True, train=True,
                     transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])),
                     batch_size=64, shuffle=False, sampler=torch.utils.data.SubsetRandomSampler(indices[:55000]))

# Build the validation loader
val_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('mnist', download=True, train=True,
                   transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])),
                   batch_size=64, shuffle=False, sampler=torch.utils.data.SubsetRandomSampler(indices[55000:]))




class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()    
        # Declare all the layers for feature extraction
        self.features = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, padding=1), 
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(in_channels=5, out_channels=10, kernel_size=3, padding=1), 
                                      nn.MaxPool2d(2, 2), nn.ReLU(inplace=True),
                                      nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(in_channels=20, out_channels=40, kernel_size=3, padding=1),
                                      nn.MaxPool2d(2, 2), nn.ReLU(inplace=True))
        # Declare all the layers for classification
        self.classifier = nn.Sequential(nn.Linear(7 *7 * 40, 1024), nn.ReLU(inplace=True),
                                       	nn.Linear(1024, 2048), nn.ReLU(inplace=True),
                                        nn.Linear(2048,10))
    def forward(self, x):
      
        # Apply the feature extractor in the input
        x = self.features(x)
        
        # Squeeze the three spatial dimensions in one
        x = x.view(-1, 7 * 7 * 40)
        
        # Classify the images
        x =self.classifier(x)
        return x

# Instantiate the network
model = Net()

# Instantiate the cross-entropy loss
criterion = nn.CrossEntropyLoss()

# Instantiate the Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.001)
