import os
import torch
import numpy as np
from PIL import Image
import pandas as pd
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from efficientdet.utils import convert_map_to_lane_map, convert_map_to_road_map

from torch.utils.data import Dataset, DataLoader

NUM_SAMPLE_PER_SCENE = 126
NUM_IMAGE_PER_SAMPLE = 6
image_names = [
    'CAM_FRONT_LEFT.jpeg',
    'CAM_FRONT.jpeg',
    'CAM_FRONT_RIGHT.jpeg',
    'CAM_BACK_LEFT.jpeg',
    'CAM_BACK.jpeg',
    'CAM_BACK_RIGHT.jpeg',
    ]


class Normalizer_test(object):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std
        

    def __call__(self, sample):
        height, width = sample.shape[1], sample.shape[2]
        
        
        tensor_mean = torch.Tensor(3, height, width)
        tensor_std = torch.Tensor(3, height, width)
        
        for i in range(3):
            tensor_mean[i].fill_(self.mean[i])
            tensor_std[i].fill_(self.std[i])
            
#         print(sample)
#         print('----')
        
        
        sample = (sample/255 - tensor_mean)/tensor_std
#         print(sample.shape)
#         print(sample)
        return sample

















