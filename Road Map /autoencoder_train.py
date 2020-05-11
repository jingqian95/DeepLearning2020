from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import transforms
from matplotlib import pyplot as plt
import argparse
import os
import random

import numpy as np
import pandas as pd

import matplotlib

matplotlib.rcParams['figure.figsize'] = [5, 5]
matplotlib.rcParams['figure.dpi'] = 200

import torch.nn.functional as F

from data_helper import UnlabeledDataset, LabeledDataset
from helper import collate_fn, draw_box
from autoencoder import get_auto_encoder

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='data')
parser.add_argument('--epoch', type=int)
parser.add_argument('--save_dir',type=str)
opt = parser.parse_args()

batch_size = 256

transform = transforms.Compose([
    transforms.ToTensor()
    # transforms.Normalize((0.5,), (0.5,))
])

image_folder = opt.data_dir
unlabeled_scene_index = np.arange(106)
#transform = torchvision.transforms.ToTensor()

dataset_size = len(unlabeled_scene_index)
indices = list(range(dataset_size))
indices = np.random.shuffle(indices)
split = int(dataset_size * 0.8 )
train_indices, val_indices = unlabeled_scene_index[:split], unlabeled_scene_index[split:]

unlabeled_trainset_train = UnlabeledDataset(image_folder=image_folder, scene_index=train_indices, first_dim='sample', transform=transform)
unlabeled_trainset_val = UnlabeledDataset(image_folder=image_folder, scene_index=val_indices, first_dim='sample', transform=transform)
trainloader = torch.utils.data.DataLoader(unlabeled_trainset_train, batch_size=1, shuffle=True, num_workers=2)
valloader = torch.utils.data.DataLoader(unlabeled_trainset_val, batch_size=1, shuffle=True, num_workers=2)

device = 'cuda'

model = get_auto_encoder().to(device)
criterion = nn.MSELoss()

# Configure the optimiser

learning_rate = 1e-3

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate,
)
# Train standard or denoising autoencoder (AE)
num_epochs = opt.epoch
model.train()
# do = nn.Dropout()  # comment out for standard AE
for epoch in range(num_epochs):
    for data in trainloader:
        img = data.view(6,3,256,306)
        img = img.to(device)
        # ===================forward=====================
        output = model.forward(img) 
        loss = criterion(output,img.data)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    print(f'epoch [{epoch + 1}/{num_epochs}], loss:{loss.item():.4f}')

torch.save(model.state_dict(), opt.save_dir)