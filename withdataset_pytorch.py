#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 16:48:05 2021

@author: nataly

MNIST- NUMEROS UNO SOLO CANAL
CIFAR-10

"""
import torch
import torchvision
import torch.utils.data
import torchvision.transforms as transforms
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

"""
Inspecting the dataloaders
Now you are going to explore a bit the dataloaders you created in the previous 
exercise. In particular, you will compute the shape of the dataset in addition 
to the minibatch size.
"""

# Compute the shape of the training set and testing set
trainset_shape = trainloader.dataset.train_data.shape
testset_shape = testloader.dataset.test_data.shape

# Print the computed shapes
print(trainset_shape,testset_shape)

# Compute the size of the minibatch for training set and testing set
trainset_batchsize = trainloader.batch_size
testset_batchsize = testloader.batch_size

# Print sizes of the minibatch
print(trainset_batchsize, testset_batchsize)