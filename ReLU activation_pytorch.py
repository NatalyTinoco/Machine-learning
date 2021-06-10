#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 15:54:32 2021

@author: nataly

ReLU activation
In this exercise, we have the same settings as the previous exercise. 
But now we are going to build a neural network which has non-linearity.
 By doing so, we are going to convince ourselves that networks with multiple
 layers and non-linearity functions cannot be expressed as a neural network
 with one layer.
"""
import torch
import torch.nn as nn

input_layer=torch.rand(4)
weight_1=torch.rand(4,4)
weight_2=torch.rand(4,4)
weight_3=torch.rand(4,1)

# Instantiate non-linearity
relu = nn.ReLU()
# Apply non-linearity on the hidden layers
hidden_1_activated = relu(torch.matmul(input_layer, weight_1))
hidden_2_activated = relu(torch.matmul(hidden_1_activated, weight_2))
print(torch.matmul(hidden_2_activated, weight_3))

# Apply non-linearity in the product of first two weights. 
weight_composed_1_activated = relu(torch.matmul(weight_1, weight_2))

# Multiply `weight_composed_1_activated` with `weight_3
weight = torch.matmul(weight_composed_1_activated, weight_3)

# Multiply input_layer with weight
print(torch.matmul(input_layer, weight))


"""
ReLU activation again
Neural networks don't need to have the same number of units in each layer. 
Here, you are going to experiment with the ReLU activation function again,
 but this time we are going to have a different number of units in the layers 
 of the neural network. The input layer will still have 4 features, but then 
 the first hidden layer will have 6 units and the output layer will have 
 2 units.
"""


# Instantiate ReLU activation function as relu
relu = nn.ReLU()
# Initialize weight_1 and weight_2 with random numbers
weight_1 = torch.rand(4, 6)
weight_2 = torch.rand(6, 2)
# Multiply input_layer with weight_1
hidden_1 = torch.matmul(input_layer, weight_1)
# Apply ReLU activation function over hidden_1 and multiply with weight_2
hidden_1_activated = relu(hidden_1)
print(torch.matmul(hidden_1_activated, weight_2))

