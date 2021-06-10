#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 31 11:15:15 2021

@author: nataly
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
NeuralNetworks-Recapimport 

You haven't created a neural network since the end of the first chapter, 
so this is a good time to build one (practice makes perfect). 
Build a class for a neural network which will be used to train on the MNIST 
dataset. The dataset contains images of shape (28, 28, 1), so you should 
deduct the size of the input layer. For hidden layer use 200 units, while 
for output layer use 10 units (1 for each class). For activation function, 
use relu in a functional way (nn.Functional is already imported as F).


"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Define the class Net
class Net(nn.Module):
    def __init__(self):    
    	# Define all the parameters of the net
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28 * 1, 200)
        self.fc2 = nn.Linear(200,10)

    def forward(self, x):   
    	# Do the forward pass
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
"""
Training a neural network
Given the fully connected neural network (called model) which you built in the 
previous exercise and a train loader called train_loader containing the MNIST 
dataset (which we created for you), you're to train the net in order to
 predict the classes of digits. You will use the Adam optimizer to optimize 
 the network, and considering that this is a classification problem you are
 going to use cross entropy as loss function.
"""   

# Instantiate the Adam optimizer and Cross-Entropy loss function
model = Net()   
optimizer = optim.Adam(model.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss()
  
for batch_idx, data_target in enumerate(trainloader):
    data = data_target[0]
    target = data_target[1]
    data = data.view(-1, 28 * 28)
    optimizer.zero_grad()

    # Complete a forward pass
    output = model(data)

    # Compute the loss, gradients and change the weights
    loss = criterion(output,target)
    loss.backward()
    optimizer.step()    
    
    
"""
Using the network to make predictions
Now that you have trained the network, use it to make predictions for the data 
in the testing set. The network is called model (same as in the previous
                                                 exercise), and the loader 
is called test_loader. We have already initialized variables total and correct
 to 0

"""

correct, total = 0, 0
predictions = []
# Set the model in eval mode
model.eval()

for i, data in enumerate(testloader, 0):
    inputs, labels = data
    
    # Put each image into a vector
    inputs = inputs.view(-1, 28*28*1)
    
    # Do the forward pass and get the predictions
    outputs = model(inputs)
    _, outputs = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (outputs == labels).sum().item()
    
print('The testing set accuracy of the network is: %d %%' % (100 * correct / total))    