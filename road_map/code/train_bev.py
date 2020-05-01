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
from data_helper import UnlabeledDataset, LabeledDataset
from helper import collate_fn, draw_box

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
def train_bev(trainloader, model, optimizer, criterion, epoch):
    model.train()
    total_loss = 0
    for i, (sample, target, road_image) in enumerate(trainloader):
        optimizer.zero_grad()
        samples = torch.stack(sample).to(device)
        samples = samples.view(samples.shape[0], -1, 256, 306).to(device)
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

def eval_bev(valloader, model, criterion):
    model.eval()
    total_loss = 0
    tp,fp,tn,fn = 0,0,0,0
    t_scores = []
    for i, (sample, target, road_image) in enumerate(valloader):
        sample, road_image = torch.stack(sample).to(device), torch.stack(road_image).float().to(device)
        sample = sample.view(sample.shape[0], -1, 256, 306)
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

if __name__ == "__main__":
    device = 'cuda'
    image_folder = '../../student_data/data'
    annotation_csv = '../../student_data/data/annotation.csv'

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

    transform = torchvision.transforms.ToTensor()
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
    trainloader = torch.utils.data.DataLoader(labeled_trainset, batch_size=3, shuffle=True, num_workers=2,
                                              collate_fn=collate_fn)
    valloader = torch.utils.data.DataLoader(labeled_valset, batch_size=3, shuffle=True, num_workers=2,
                                            collate_fn=collate_fn)

    _C = CN()

    _C.OUTPUT_DIR = ''
    _C.LOG_DIR = ''
    _C.GPUS = (0,)
    _C.WORKERS = 4
    _C.PRINT_FREQ = 20
    _C.AUTO_RESUME = False
    _C.PIN_MEMORY = True
    _C.RANK = 0

    # Cudnn related params
    _C.CUDNN = CN()
    _C.CUDNN.BENCHMARK = True
    _C.CUDNN.DETERMINISTIC = False
    _C.CUDNN.ENABLED = True

    # common params for NETWORK
    _C.MODEL = CN()
    _C.MODEL.NAME = 'seg_hrnet'
    _C.MODEL.PRETRAINED = ''
    _C.MODEL.EXTRA = CN(new_allowed=True)

    _C.LOSS = CN()
    _C.LOSS.USE_OHEM = False
    _C.LOSS.OHEMTHRES = 0.9
    _C.LOSS.OHEMKEEP = 100000
    _C.LOSS.CLASS_BALANCE = True

    # DATASET related params
    _C.DATASET = CN()
    _C.DATASET.ROOT = ''
    _C.DATASET.NUM_CLASSES = 1
    _C.DATASET.EXTRA_TRAIN_SET = ''

    # training
    _C.TRAIN = CN()

    _C.TRAIN.IMAGE_SIZE = [1024, 512]  # width * height
    _C.TRAIN.BASE_SIZE = 2048
    _C.TRAIN.DOWNSAMPLERATE = 1
    _C.TRAIN.FLIP = True
    _C.TRAIN.MULTI_SCALE = True
    _C.TRAIN.SCALE_FACTOR = 16

    _C.TRAIN.LR_FACTOR = 0.1
    _C.TRAIN.LR_STEP = [90, 110]
    _C.TRAIN.LR = 0.01
    _C.TRAIN.EXTRA_LR = 0.001

    _C.TRAIN.OPTIMIZER = 'sgd'
    _C.TRAIN.MOMENTUM = 0.9
    _C.TRAIN.WD = 0.0001
    _C.TRAIN.NESTEROV = False
    _C.TRAIN.IGNORE_LABEL = -1

    _C.TRAIN.BEGIN_EPOCH = 0
    _C.TRAIN.END_EPOCH = 484
    _C.TRAIN.EXTRA_EPOCH = 0

    _C.TRAIN.RESUME = False

    _C.TRAIN.BATCH_SIZE_PER_GPU = 32
    _C.TRAIN.SHUFFLE = True
    # only using some training samples
    _C.TRAIN.NUM_SAMPLES = 0

    # testing
    _C.TEST = CN()

    _C.TEST.IMAGE_SIZE = [2048, 1024]  # width * height
    _C.TEST.BASE_SIZE = 2048

    _C.TEST.BATCH_SIZE_PER_GPU = 32
    # only testing some samples
    _C.TEST.NUM_SAMPLES = 0

    _C.TEST.MODEL_FILE = ''
    _C.TEST.FLIP_TEST = False
    _C.TEST.MULTI_SCALE = False
    _C.TEST.SCALE_LIST = [1]

    # debug
    _C.DEBUG = CN()
    _C.DEBUG.DEBUG = False
    _C.DEBUG.SAVE_BATCH_IMAGES_GT = False
    _C.DEBUG.SAVE_BATCH_IMAGES_PRED = False
    _C.DEBUG.SAVE_HEATMAPS_GT = False
    _C.DEBUG.SAVE_HEATMAPS_PRED = False
    ##update config##
    _C.defrost()
    _C.merge_from_file("scratch.yml")
    _C.freeze()

    config = _C
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
        train_bev(trainloader, model, optimizer, criterion, epoch)
        val_loss,val_ts,avg_ts, pred = eval_bev(valloader, model, criterion)
        # if epoch == num_epoch - 1:
        #     print(pred)
        print("Val loss is {:.6f}, threat score is {:.6f}, average threat score is {:.6}".format(val_loss, val_ts,avg_ts))
    torch.save(model.state_dict(), save_model_dir)