import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CIFAR_CNN(nn.Module):
    def __init__(self, normalizer, dropout=0.0):
        super(CIFAR_CNN, self).__init__()
        self.normalizer = normalizer

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64*8*8, 256)
        self.fc2 = nn.Linear(256, 256)
        self.linear = nn.Linear(256, 10)
        self.dropout = nn.Dropout(p=dropout)


    def calc_representation(self, x):
        x = self.normalizer(x)
        x = F.relu(self.conv1(x))
        x = self.max_pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.max_pool(F.relu(self.conv4(x)))
        x = F.relu(self.fc1(x.view(x.size(0),64*8*8)))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return x

    def forward(self, x):
        x = self.calc_representation(x)
        x = self.linear(x)
        return x
