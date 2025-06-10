'''
Datesets and Dataloaders

If we compute datasets' code and model's code at the same time, there will be a mess.
So we can decouple each other by using "torch.utils.data.DataLoader" and "torch.utils.data.Dataset".
'''
from learning_20250608_01 import training_data

'''
1.Loading a Dataset

We use the Fashion-MNIST dataset to show how to load(there are some parameters):
a."root" is the path where the train or test data is stored
b."train" specifies training or test dataset
c."download = True" downloads the data from the internet if it's not available at root
d."transform" and "target_transform" specify the feature and label transformations
'''

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

training_data = datasets.FashionMNIST(
    root = "data",
    train = True,
    download = True,
    transform = ToTensor()
)

test_data = datasets.FashionMNIST(
    root = "data",
    train = False,
    download = True,
    transform = ToTensor()
)

'''
2.iterating and visualizing the Dataset

We can index Datasets manually like a list: training_data[index],
and use matplotlib to visualize some samples in our training data.
'''

labels_map = {
    0:"T-Shirt",
    1:"Trouser",
    2:"Pullover",
    3:"Dress",
    4:"Coat",
    5:"Sandal",
    6:"Shirt",
    7:"Sneaker",
    8:"Bag",
    9:"Ankle Boot",
}
figure = plt.figure(figsize=(8,8))
cols,rows = 3,3
for i in range(1,cols * rows+1):
    sample_idx = torch.randint(len(training_data),size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(),camp="gray")
plt.show()

'''
3.Creating a custom Dataset for files

We need 3 functions to do:"__init__", "__len__", "__getitem__".
'''

# there are a FashionMNIST dataset
import os
import pandas as pd
from torchvision.io import decode_image

class CustomImageDataset(Dataset):
    # "__init__" function will:
    # 1.initialize the path where store the image
    # 2.initialize the path of annotations files
    # 3.initialize the image transformation and label transformation
    def __init__(self,annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    # "__len__" function returns the number of samples in dataset
    def __len__(self):
        return len(self.img_labels)

    # "__getitem__" function loads and return a sample from the dataset at the given index "idx",it will:
    # 1.locate the image files' address
    # 2.turn the image to the tensor by using "decode_image"
    # 3.gain the labels from CSV data by using "self.img_labels"
    # 4.call the transform function
    # 5.return a tuple which contains tensor images and their labels
    def __getitem__(self,idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx,0])
        image = decode_image(img_path)
        label = self.img_labels.iloc[idx,1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

'''
4.Preparing data for training with DataLoaders

Dataset every times only search one sample in dataset(including feature and label),so when we want to train it,we need to:
a.passing samples in minibatch
b.in every epoch, reshuffle data to reduce model overfitting 
c.using python's "multiprocessing" to speed up data retrieval 
'''

from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data,batch_size=64,shuffle=True)
test_dataloader = DataLoader(test_data,batch_size=64,shuffle=True)

'''
5.A case
'''

# Display image and label
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")


