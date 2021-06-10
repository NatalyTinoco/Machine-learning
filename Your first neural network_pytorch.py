#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 15:22:41 2021

@author: nataly
"""
import torch
"""
Basica neuralnetwork
"""
input_layer=torch.rand(784)
# Initialize the weights of the neural network
weight_1 = torch.rand(784, 28)
weight_2 = torch.rand(28,10)

# Multiply input_layer with weight_1
hidden_1 = torch.matmul(input_layer, weight_1)

# Multiply hidden_1 with weight_2
output_layer = torch.matmul(hidden_1,weight_2)
print(output_layer)

"""
Your first PyTorch neural network
"""
  
import torch.nn as nn   
    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Instantiate all 2 linear layers  
        self.fc1 = nn.Linear(784,200)
        self.fc2 = nn.Linear(200,10)

    def forward(self, x):
      
        # Use the instantiated layers and return x
        x = self.fc1(x)
        x = self.fc2(x)
        return x
        
"""
con función RELU
We are going to convince ourselves that networks with multiple layers
 which do not contain non-linearity can be expressed as neural networks
 with one layer.
"""

input_layer=torch.rand(4)
weight_1=torch.rand(4,4)
weight_2=torch.rand(4,4)
weight_3=torch.rand(4,1)
# Calculate the first and second hidden layer
hidden_1 = torch.matmul(input_layer, weight_1)
hidden_2 = torch.matmul(hidden_1, weight_2)

# Calculate the output
print(torch.matmul(hidden_2, weight_3))

# Calculate weight_composed_1 and weight
weight_composed_1 = torch.matmul(weight_1, weight_2)
weight = torch.matmul(weight_composed_1, weight_3)

# Multiply input_layer with weight
print(torch.matmul(input_layer, weight))
