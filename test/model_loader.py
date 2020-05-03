"""
You need to implement all four functions in this file and also put your team info as a variable
Then you should submit the python file with your model class, the state_dict, and this file
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from seg_hrnet import get_seg_model
from config import config
from config import update_config


# import your model class
# import ...

# Put your transform function here, we will use it for our dataloader
# For bounding boxes task
def get_transform_task1(): 
    return torchvision.transforms.ToTensor()
# For road map task
def get_transform_task2(): 
    return torchvision.transforms.ToTensor()

class ModelLoader():
    # Fill the information for your team
    team_name = 'team_name'
    team_number = 1
    round_number = 1
    team_member = []
    contact_email = '@nyu.edu'

    def __init__(self, model_file=['put_your_model_file(or files)_name_here_objDection','lr5_epoch20_final.pt']):
        # You should 
        #       1. create the model object
        #       2. load your state_dict
        #       3. call cuda()
        # self.model = ...
        self.device = 'cuda'
        self.RoadMap_model_path = model_file[-1]
        #print("path is ",self.RoadMap_model_path)
        update_config(config)
        self.RoadMap_model = get_seg_model(config)
        self.RoadMap_model.load_state_dict(torch.load(self.RoadMap_model_path))
        self.RoadMap_model = self.RoadMap_model.to(self.device)

    def get_bounding_boxes(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a tuple with size 'batch_size' and each element is a cuda tensor [N, 2, 4]
        # where N is the number of object

        return torch.rand(1, 15, 2, 4) * 10

    def get_binary_road_map(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a cuda tensor with size [batch_size, 800, 800] 
        self.RoadMap_model.eval()
        samples = samples.view(1, 18, 256, 306).to(self.device)
        pred = self.RoadMap_model(samples)
        output = pred > 0.5
        # print(pred.shape)
        return output
