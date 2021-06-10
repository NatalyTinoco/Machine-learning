#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 31 18:57:55 2021

@author: nataly

Convolution operator - OOP way
Let's kick off this chapter by using convolution operator from the torch.nn 
package. You are going to create a random tensor which will represent your 
image and random filters to convolve the image with. Then you'll apply those 
images.

The torch library and the torch.nn package have already been imported for you.
"""
import torch
import torch.nn
# Create 10 random images of shape (1, 28, 28)
images = torch.rand(10, 1, 28, 28)

# Build 6 conv. filters
conv_filters = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=1, padding=1)

# Convolve the image with the filters
output_feature = conv_filters(images)
print(output_feature.shape)


"""
Convolution operator - Functional way
While I and most of PyTorch practitioners love the torch.nn package (OOP way), 
other practitioners prefer building neural network models in a more functional w
ay, using torch.nn.functional. More importantly, it is possible to mix the 
concepts and use both libraries at the same time (we have already done it in 
                                                  the previous chapter). 
You are going to build the same neural network you built in the previous 
exercise, but this time using the functional way.

As before, we have already imported the torch library and torch.nn.functional 
as F.

"""
import torch.nn.functional as F

# Create 10 random images
image = torch.rand(10,1, 28, 28)

# Create 6 filters
filters = torch.rand(6,1,3,3)

# Convolve the image with the filters
output_feature = F.conv2d(image,filters,stride=1, padding=1)
print(output_feature.shape)