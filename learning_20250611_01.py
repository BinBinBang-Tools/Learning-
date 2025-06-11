'''
Build a Neural Network

Neural networks comprise layers/modules that perform operations on data.
The "torch.nn" namespace provides all the buildingblocks, every module in PyTorch subclasses the "nn.module".
'''

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

'''
1.select device

We can choose use CUDA to train model,if not,cpu is also a choice.
'''

device = (
    "cuda" if torch.cuda.is_available()
    # else "mps" if torch.backends.mps.is_available()  # a more choice in Apple Silicon
    else "cpu"
)
print(f"Using {device} device")

'''
2.Define the class

By using the "__init__" and "__forward__" function, with the nn.Module.

'''

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28,512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )


    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# ensurn our data process in CUDA
model = NeuralNetwork().to(device)
print(model)

# when we need to use the model, we only pass the data wo it.
# Don't call "model.forward" directly!

# Calling the model on the input data will return a 2-dimanshion tensor,
# dim = 0 :10 original predictions(output)
# dim = 1 :true values

# one case (using "nn.softmax")
X = torch.rand(1,28,28,device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class:{y_pred}")

'''
3.Model layers

There are some kinds of layers:
a.nn.Flatten
b.nn.Linear
c.nn.ReLU
d.nn.Sequential
e.nn.Softmax

eg:FashionMNIST
'''

# using a case to show:a minibatch which contains 3 28*28 images
input_image = torch.rand(3,28,28)
print(input_image.size)

# "nn.Flatten"
# initialize the "nn.Flatten" layer to convert each 2D 28*28 image into a contiguous array of 784 pixel  values,
# and maintains the dim=0
# [3,1,28,28] to [3,784]

flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())

# "nn.Linear"
# using stored weights("W") and biases("b") to apply a linear transformation on the input data
layer1 = nn.Linear(in_features=28*28,out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())

# "nn.ReLU"
# Non-linear activations are what create the complex mappings between the input and output.
# usually applied after linear transformations to introduce nonlinearity
print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")

# "nn.Sequential"
# a container,in which data will be process in order
# eg: seq_modules
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20,10)
)
input_image = torch.rand(3,28,28)
logits = seq_modules(input_image)

# "nn.Softmax"
# the last layer will return the logits,we put the logits to the nn.Softmax,they will be scaled to [0,1]
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)

'''
4.Model Parameters

Many layers inside a neural network are parameterized,they own many weights and biases.
By using nn.Module, they can be catched and we can using "parameters()" and "named_parameters()" to get these parameters.
'''

print(f"Model structure: {model}\n\n")

for name,param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]}\n")