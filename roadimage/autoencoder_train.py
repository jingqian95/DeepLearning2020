import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from matplotlib import pyplot as plt

import os
import random

import numpy as np
import pandas as pd
import argparse
import matplotlib

matplotlib.rcParams['figure.figsize'] = [5, 5]
matplotlib.rcParams['figure.dpi'] = 200

import torch.nn.functional as F

from data_helper_2 import UnlabeledDataset, LabeledDataset
from helper import collate_fn, draw_box

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
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
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
trainloader = torch.utils.data.DataLoader(unlabeled_trainset_train, batch_size=2, shuffle=True, num_workers=2)
valloader = torch.utils.data.DataLoader(unlabeled_trainset_val, batch_size=2, shuffle=True, num_workers=2)

device = 'cuda'
d = 30  # for standard AE (under-complete hidden layer)

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(256*306, d),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(d, 256*306),
            nn.Tanh(),
        )
        self.d = d
    def forward(self, x):
        x = self.encoder(x.view(-1, 256*306))
        x = self.decoder(x)
        return x
    
model = Autoencoder().to(device)
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
        img, _ = data
        img = img.to(device)
        # noise = do(torch.ones(img.shape))
        # img_bad = (img * noise).to(device)  # comment out for standard AE
        # ===================forward=====================
        output = model.forward(img)  # feed <img> (for std AE) or <img_bad> (for denoising AE)
        loss = criterion(output.view(img.data.shape[0],img.data.shape[1],256,306), img.data)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    print(f'epoch [{epoch + 1}/{num_epochs}], loss:{loss.item():.4f}')

torch.save(model.state_dict(), opt.save_dir)