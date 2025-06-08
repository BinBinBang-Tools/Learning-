'''
Quick start
Learn the apis of the pytorch
'''
from pickletools import optimize

'''
1.Data

Pytorch has 2 blocks(or named primitives,means a series of operations which can not be stopped) about the data
a.torch.utils.data.DataLoader:
    wraps an iterable around the Dataset
b.torch.utils.data.Dataset:
    stores the samples and their corresponding labels

There are many libraries which contains many datasets:TorchText,TorchVision,TorchAudio and so on.
We can downloads datasets from these libraries.
'''
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# download training data from open datasets,for example:FashionMNIST
training_data = datasets.FashionMNIST(
    root = "data",
    train = True,
    download = True,
    transform = ToTensor(),
)

# download test data from open datasets,for example:FashionMNIST
test_data = datasets.FashionMNIST(
    root = "data",
    train = False,
    download = True,
    transform = ToTensor(),
)

# pass the Dataset as an  argument  to DataLoader,set the batch_size=64
batch_size=64 #one epoch will train 64 data
# Creat data loaders
train_dataloader = DataLoader(training_data,batch_size=batch_size)
test_dataloader = DataLoader(test_data,batch_size=batch_size)

for X,y in test_dataloader:
    print(f"Shape of X[N,C,H,W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

'''
2.Models

When we use neural network,we need to define the structure at first.
And when we define the structure,we need create a class that inherits from nn.Module.

There are two demands when we define a Module.
a.use the __init__ function define the layers constructure
b.use the forward function pass the data from one layer to another layer

To reduce the time of training,we will use CUDA,MPS,MTIA,XPU,and so on.
If not we can use cpu
'''

# Detect and use GPU
device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Define Module
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10),
        )

    def forward(self,x):
        x=self.flatten(x)
        logits=self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

'''
3.Optimizing

To train a model,we need a loss function and an optimizer

In every single training loop,the model makes predictions on the training dataset(fed to it in batches),
and backpropagates the prediction error to adjust the model's parameters.
Besides training, we also check the model's performance against the test dataset to ensure it is learning.
And we will print every batch to ensure the prediction up and loss down.
'''
loss_fn = nn.CrossEntropyLoss() #loss and optimize
optimizer = torch.optim.SGD(model.parameters(),lr=1e-3)

# forward and back,training
def train(dataloader,model,lloss_fn,optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch,(X,y) in enumerate(dataloader):
        X,y = X.to(device), y.to(device)

        # compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch%100 == 0:
            loss,current = loss.item(),(batch + 1) * len(X)
            print(f"loss:{loss:>7f} [{current:>5d}/{size:>5d}]")

# test
def test(dataloader,model,loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss,correct = 0,0
    with torch.no_grad():
        for X,y in dataloader:
            X,y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}\n")

# training 5 times
epochs = 5
for t in range(epochs):
    print(f"Epoch{t+1}\n---------------------------")
    train(train_dataloader,model,loss_fn,optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

'''
4.Saving

After the training and test, if the results are great,we can save the model to help other projects by
serializing the state dictionary(containing the model parameters). 
'''

torch.save(model.state_dict(),"model.pth")
print("Saved PyTorch Model State to model.pth")

'''
5.Loading 

When we need to use the model, we need to play 2 steps:
a.re-creating the model structure
b.loading the state dictionary into the structure 
'''

model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.path",weights_only=True))

# a case about such codes
classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
]

model.eval()
x,y= test_data[0][0],test_data[0][1]
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted,actual = classes[pred[0].argmax(0),classes[y]]
    print(f'Predicted:"{predicted}",Actual:{actual}')