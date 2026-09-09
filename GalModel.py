import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset

side = 64   # new pixel length

class GalCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size= 3, stride = 1, padding = 1, bias = True)      # convolution layer 1
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1, bias = True)     # convolution layer 2
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1, bias = True)     # convolution layer 3

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        self.relu = nn.ReLU()           # activation ReLU function
        self.pool = nn.MaxPool2d(2)     # max pooling 2x2
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(8192, 128)     # 3d input from first conv layer
        self.fc2 = nn.Linear(128, 8)        # 8 possible outputs (x, 8)
        self.dropout = nn.Dropout(0.4)     # neglect n percent of neurons


    def forward(self, x):

        # first layer
        x = self.conv1(x)   # first convolution layer
        x = self.bn1(x)     # conv layer normalization
        x = self.relu(x)    # activation
        x = self.pool(x)    # pool 2x2

        # second layer
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)

        # third layer
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)

        # output layer
        x = self.flatten(x)    # flatten from tensor to list

        x = self.fc1(x)     # first neural connection
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)     # second neural connection

        return x


class GalDataset(Dataset):

    def __init__(self, hf_dataset):

        self.data = hf_dataset


    def __len__(self):

        return len(self.data)


    def __getitem__(self, index):   # load image

        sample = self.data[index]
        image = prep_img(sample["image"])
        label = sample["label"]

        return image, label


def prep_img(image):

    img = image.convert('L')    # convert to BW
    resize_img = img.resize((side, side), Image.Resampling.LANCZOS)  # resize w filter
    pixel_array = np.array(resize_img, dtype = np.float32)      # create pixel list

    pixel_tensor = torch.tensor(pixel_array)
    pixel_tensor = pixel_tensor.unsqueeze(0)   # changes 2d tensor to match torch 4d tensor

    pixel_tensor = pixel_tensor.float() / 255.0    # normalize

    return(pixel_tensor)