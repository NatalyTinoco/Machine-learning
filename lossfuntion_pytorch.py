#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 16:06:19 2021

@author: nataly
Loss Functions

1. Initialize neural networks with random weights.
2. Do a forward pass.
3. Calculate loss function (1 number).
4. Calculate the gradients.
5. Change the weights based on gradients.

    * For regression: least squared loss.
    *For classication: somax cross-entropy loss.
    *For more complicated problems (like object detection), 
     more complicated losses
"""
import torch
import torch.nn as nn

"""
1. Calculating loss function in PyTorch
You are going to code the previous exercise, and make sure that we computed the
 loss correctly. Predicted scores are -1.2 for class 0 (cat), 0.12 for class 1 
 (car) and 4.8 for class 2 (frog). The ground truth is class 2 (frog). 
 Compute the loss function in PyTorch.
 
CrossEntropyLoss
"""

# Initialize the scores and ground truth
logits = torch.tensor([[-1.2,0.12,4.8]])
ground_truth = torch.tensor([2])

# Instantiate cross entropy loss
criterion = nn.CrossEntropyLoss()

# Compute and print the loss
loss = criterion(logits,ground_truth)
print(loss)


"""
Loss function of random scores
"""

# Initialize logits and ground truth
logits = torch.rand(1,1000)
ground_truth = torch.tensor([111])

# Instantiate cross-entropy loss
criterion=nn.CrossEntropyLoss()

# Calculate and print the loss
loss = criterion(logits,ground_truth)
print(loss)


