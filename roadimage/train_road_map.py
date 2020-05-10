import argparse
import os
import pprint
import shutil
import sys
import random
import json

import logging
import time
import timeit
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim

from seg_hrnet import get_seg_model
from torchvision import transforms

import os
import random

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.figsize'] = [3, 3]
matplotlib.rcParams['figure.dpi'] = 200

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from helper import collate_fn, draw_box
from data_helper_2 import UnlabeledDataset, LabeledDataset
from helper import collate_fn, draw_box
from config import config
from config import update_config
import torchvision.models as models
import torch.utils.model_zoo as model_zo
import torchvision
#!pip install yacs
from yacs.config import CfgNode as CN

def threat_score(tp,fp,tn,fn,pred,road_image):
    output_class = (pred > 0.5).float()
    ##tp is where prediction class = road_image class = 1
    tp_tmp = torch.sum(torch.mul(output_class, road_image))
    ##fp is where prediction class = 1 and road_image = 0
    fp_tmp = ((output_class - road_image) == 1).sum()
    ###tn is where prediction class = road_image class = 0
    tn_tmp = ((output_class - road_image) == 0).sum() - tp_tmp
    ## fn is where prediction class = 0 and road_image = 1
    fn_tmp = ((output_class - road_image) == -1).sum()
    tp += tp_tmp
    fp += fp_tmp
    tn += tn_tmp
    fn += fn_tmp
    tmp_ts = tp_tmp / (tp_tmp + fp_tmp + fn_tmp)
    return tp,fp,tn,fn,tmp_ts
def train_bev(trainloader, model, optimizer, criterion, epoch,auto_encoder):
    model.train()
    total_loss = 0
    for i, (sample, target, road_image) in enumerate(trainloader):
        optimizer.zero_grad()
        if auto_encoder:
            tmp_sample = []
            for s in sample:
                s = s.to(device)
                tmp_sample.append(auto_encoder(s))
            samples = torch.stack(tmp_sample).to(device)
        else:
            samples = torch.stack(sample).to(device)
        samples = samples.view(samples.shape[0], -1, 256, 306).to(device)
        #print('!!!!!!!!!!!!!!!!!!!!sample shape is!!!!!!!!!!!!!!',samples.shape)
        road_image = torch.stack(road_image).type(dtype=torch.float32).to(device)
        pred = model.forward(samples)
        loss = criterion(pred, road_image)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        if i != 0 and i % 5 == 0:
            avg_loss = total_loss / (i + 1)
            print('Epoch: {} | Avg Loss: {}'.format(epoch, avg_loss))
    avg_loss = total_loss / len(trainloader)
    print('Trained Epoch {} | Total Avg Loss: {}'.format(epoch, avg_loss))

def eval_bev(valloader, model, criterion,auto_encoder):
    model.eval()
    total_loss = 0
    tp,fp,tn,fn = 0,0,0,0
    t_scores = []
    for i, (sample, target, road_image) in enumerate(valloader):
        if auto_encoder:
            tmp_sample = []
            for s in sample:
                s = s.to(device)
                tmp_sample.append(auto_encoder(s))
            sample = torch.stack(tmp_sample).to(device)
        else:
            sample = torch.stack(sample).to(device)
        road_image =  torch.stack(road_image).float().to(device)
        sample = sample.view(sample.shape[0], -1, 256, 306)
       # print('!!!!!!!!!sample shape is',sample.shape)
        with torch.no_grad():
            pred = model(sample)
        # if len(valloader) - i <= 5:
        #     pred_file[i] = pred.cpu().numpy().tolist()
        loss = criterion(pred, road_image)
        total_loss += loss.item()
        tp,fp,tn,fn,tmp_ts = threat_score(tp,fp,tn,fn,pred,road_image)
        t_scores.append(tmp_ts)
        # preds.append(pred.cpu().numpy())
    # print(pred_file)
    # with open('pred.txt', 'w') as outfile:
    #     json.dump(pred_file, outfile)
    print("threat scores are :", t_scores)
    loss = total_loss / len(valloader)
    ts = tp / (tp + fp + fn)
    return loss, ts, sum(t_scores)/len(t_scores),pred


class ResNetMultiImageInput(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    def __init__(self, block, layers, num_classes=1000, num_input_images=1):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1):
        """Constructs a ResNet model.
        Args:
            num_layers (int): Number of resnet layers. Must be 18 or 50
            pretrained (bool): If True, returns a model pre-trained on ImageNet
            num_input_images (int): Number of frames stacked as input
        """
        assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
        blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
        block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
        model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images)

        if pretrained:
            loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
            loaded['conv1.weight'] = torch.cat(
                [loaded['conv1.weight']] * num_input_images, 1) / num_input_images
            model.load_state_dict(loaded)
        return model

class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained, num_input_images=1):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)
        else:
            self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features


class AutocoderModel(nn.Module):
    def __init__(self):
        super(AutocoderModel, self).__init__()
        self.encoder = ResnetEncoder(num_layers=18, pretrained=False)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 128, 10),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 10),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 10),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=6, stride=2),
            nn.BatchNorm2d(3)
        )

    def forward(self, x):
        bt_sz = x.size(0)
        x = self.encoder(x)[-1]
        x = self.decoder(x)
        out = F.interpolate(x, (256, 306))
        return out

if __name__ == "__main__":
    device = 'cuda'
    image_folder = '/scratch/jq689/deeplearning/student_data/data'
    annotation_csv = '/scratch/jq689/deeplearning/student_data/data/annotation.csv'

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0);
    # You shouldn't change the unlabeled_scene_index
    # The first 106 scenes are unlabeled
    unlabeled_scene_index = np.arange(106)
    # The scenes from 106 - 133 are labeled
    # You should devide the labeled_scene_index into two subsets (training and validation)
    labeled_scene_index = np.arange(106, 134)

    dataset_size = len(labeled_scene_index)
    indices = list(range(dataset_size))
    indices = np.random.shuffle(indices)
    split = int(dataset_size * 0.8)
    train_indices, val_indices = labeled_scene_index[:split], labeled_scene_index[split:]

    transform = transforms.Compose([
        transforms.ToTensor()
        #transforms.Normalize((0.5,), (0.5,))
    ])
    labeled_trainset = LabeledDataset(image_folder=image_folder,
                                      annotation_file=annotation_csv,
                                      scene_index=train_indices,
                                      transform=transform,
                                      extra_info=False
                                      )

    labeled_valset = LabeledDataset(image_folder=image_folder,
                                    annotation_file=annotation_csv,
                                    scene_index=val_indices,
                                    transform=transform,
                                    extra_info=False
                                    )
    trainloader = torch.utils.data.DataLoader(labeled_trainset, batch_size=1, shuffle=True, num_workers=2,
                                              collate_fn=collate_fn)
    valloader = torch.utils.data.DataLoader(labeled_valset, batch_size=1, shuffle=True, num_workers=2,
                                            collate_fn=collate_fn)
    auto_encoder = AutocoderModel().to(device)
    PATH = '/scratch/jq689/deeplearning/DeepLearning2020/code/exp_models/autocoder_complex_epoch10.pt'
    auto_encoder.load_state_dict(torch.load(PATH))

    update_config(config)
    model = get_seg_model(config)
    model = model.to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD([{'params':
                                      filter(lambda p: p.requires_grad,
                                             model.parameters()),
                                  'lr': config.TRAIN.LR}],
                                lr=config.TRAIN.LR,
                                momentum=config.TRAIN.MOMENTUM,
                                weight_decay=config.TRAIN.WD,
                                nesterov=config.TRAIN.NESTEROV,
                                )
    num_epoch = int(sys.argv[1])
    save_model_dir = sys.argv[2]
    print("number of epoch :", num_epoch,"learning rate is",config.TRAIN.LR)
    t_scores = []
    for epoch in range(num_epoch):
        train_bev(trainloader, model, optimizer, criterion, epoch,auto_encoder)
        val_loss,val_ts,avg_ts, pred = eval_bev(valloader, model, criterion,auto_encoder)
        # if epoch == num_epoch - 1:
        #     print(pred)
        print("Val loss is {:.6f}, threat score is {:.6f}, average threat score is {:.6}".format(val_loss, val_ts,avg_ts))
        torch.save(model.state_dict(), save_model_dir)