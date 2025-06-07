'''
Defining a Neural Network in PyTorch(Using MNIST dataset)

Main steps
1.import all necessary libraries for loading our data
2.define and initialize the neural network
3.specify how data will pass through your model
4.pass data through your model to test
'''
from torch.nn.functional import layer_norm

'''
1.import
'''
import torch
import torch.nn as nn   #blocks,eg:nn.linear,nn.Conv2d,nn.LSTM,nn.Module,nn.Loss
import torch.nn.functional as F  #function call,eg:F.relu(),F.sigmoid(),F.conv2d

'''
2.define
'''
class Net(nn.Module):    #the first demand of defining a model(there are two in total)
    def __init__(self):
        super(Net,self).__init__()

        # first 2D convolutionnal layer, taking in 1 put channel (image)
        # outputting 32 convokutional features, with a square kernael size of 3
        self.conv1 = nn.Conv2d(1,32,3,1)
        # second 2D convolutionnal layer, taking in 32 input layers
        # outputting 64 convokutional features, with a square kernael size of 3
        self.conv2 = nn.Conv2d(32, 64, 3, 1)

        # designed to ensure that adjacent pixels are rither all 0s or all active
        # with an input probability
        self.dropout1 = nn.Dropout2d(0.25)
        # dropout2 maybe is a 1 demantion but the doc is 2 demansion
        # self.dropout2 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout(0.5)


        # first fully connected layer
        self.fc1 = nn.Linear(9216,128)
        # second fully connected layer that outputs our 10 labels
        self.fc2 = nn.Linear(128, 10)

my_nn = Net()
print(my_nn)

'''
3.pass
'''

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1,32,3,1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        # dropout2 maybe is a 1 demantion but the doc is 2 demansion
        # self.dropout2 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216,128)
        self.fc2 = nn.Linear(128, 10)

    # x reprsents data
    def forward(self,x):    #the second demand of defining a model(there are two in total)
        # pass data through  conv1
        x = self.conv1(x)
        # use the rectified-linear activation function over x
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        # run max pooling over x
        x = F.max_pool2d(x,2)
        # pass data through dropout1
        x = self.dropout1(x)
        # flatten x witn start_dim=1
        x = torch.flatten(x,1)
        # pass data through fc1
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        # apply softmax to x
        output = F.log_softmax(x,dim=1)
        return output

'''
4.test
'''

# use some random data to test
random_data = torch.rand((1,1,28,28))
my_nn = Net()
result = my_nn(random_data)
print(result)