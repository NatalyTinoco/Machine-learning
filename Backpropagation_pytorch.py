#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 13:26:38 2021

@author: nataly
"""
import torch
# Initialize x, y and z to values 4, -3 and 5
x = torch.tensor(4., requires_grad=True)
y = torch.tensor(-3., requires_grad=True)
z = torch.tensor(5.,requires_grad=True)

# Set q to sum of x and y, set f to product of q with z
q =x+y
f = q*z

# Compute the derivatives
f.backward()

# Print the gradients
print("Gradient of x is: " + str(x.grad))
print("Gradient of y is: " + str(y.grad))
print("Gradient of z is: " + str(z.grad))
###############################################
"""
Calculating gradients in PyTorch

"""
# Multiply tensors x and y
q = x*y

# Elementwise multiply tensors z with q
f = z*q

mean_f = torch.mean(f)

# Calculate the gradients
mean_f.backward()
