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

import yaml
import pandas as pd
from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess_test, invert_affine, postprocess
from test_transformer.obj_det_utils import Normalizer_test
from torchvision import transforms
import numpy as np


# import your model class
# import ...

# Put your transform function here, we will use it for our dataloader
# For bounding boxes task
def get_transform_task1(): 
    to_tensor = torchvision.transforms.ToTensor()
    normalizer = Normalizer_test()
    normalizer_2 = torchvision.transforms.Normalize(mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229))
    transform = transforms.Compose([to_tensor, normalizer_2])

    return transform
# For road map task
def get_transform_task2(): 
    return torchvision.transforms.ToTensor()

class ModelLoader():
    # Fill the information for your team
    team_name = 'Thief Vegetable'
    team_number = 11
    round_number = 2
    team_member = ['mj1477','jq689','yz5336']
    contact_email = 'jq689@nyu.edu'

    def __init__(self, model_file=['weights/best-efficientdet-d2_21909_val.pth','weights/lr5_epoch20_final.pt']):
        # You should 
        #       1. create the model object
        #       2. load your state_dict
        #       3. call cuda()
        # self.model = ...
        self.device = 'cuda'
        self.obj_det_weight = model_file[0]
        project_name = 'dl2020'
        params = yaml.safe_load(open(f'projects/{project_name}.yml'))
        obj_list = params['obj_list']
        self.input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]

        self.compound_coef = int(self.obj_det_weight.split('-d')[1][0])
        self.nms_threshold = 0.5
        self.threshold = 0.05

        self.obj_det_model = EfficientDetBackbone(compound_coef=self.compound_coef, num_classes=len(obj_list),
                                     ratios=eval(params['anchors_ratios']), scales=eval(params['anchors_scales']))

        
        self.obj_det_model.load_state_dict(torch.load(self.obj_det_weight, map_location=torch.device(self.device)))#, map_location=self.obj_det_model.device()))
        self.obj_det_model.requires_grad_(False)
        self.obj_det_model.eval()
        self.obj_det_model = self.obj_det_model.to(self.device)
        
        #print("path is ",self.RoadMap_model_path)

        self.RoadMap_model_path = model_file[-1]
        update_config(config)
        self.RoadMap_model = get_seg_model(config)
        self.RoadMap_model.load_state_dict(torch.load(self.RoadMap_model_path))
        self.RoadMap_model = self.RoadMap_model.to(self.device)

    def get_bounding_boxes(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a tuple with size 'batch_size' and each element is a cuda tensor [N, 2, 4]
        # where N is the number of object

        results = pd.DataFrame({'category_id': [],
                            'score': [],
                            'bbox': []})
        columns = ['category_id', 'score', 'bbox']

        regressBoxes = BBoxTransform()
        # use to clip the boxes to 0, width/height
        clipBoxes = ClipBoxes()
        samples = samples.cpu()
        ori_sample, framed_img, framed_meta = preprocess_test(samples, max_size = self.input_sizes[self.compound_coef])

        sample = torch.from_numpy(framed_img)
        sample = sample.cuda()
        sample = sample.unsqueeze(0).permute(0, 3, 1, 2)


        features, regression, classification, anchors = self.obj_det_model(sample)
    
        preds = postprocess(sample,
                            anchors, regression, classification,
                            regressBoxes, clipBoxes,
                            self.threshold, self.nms_threshold)
    

        preds = invert_affine([framed_meta], preds)[0]

        scores = preds['scores']
        class_ids = preds['class_ids']
        rois = preds['rois']


        if rois.shape[0] > 0:
            # x1,y1,x2,y2 -> x1,y1,w,h
            rois[:, 2] -= rois[:, 0]
            rois[:, 3] -= rois[:, 1]

            bbox_score = scores

            for roi_id in range(rois.shape[0]):
                score = float(bbox_score[roi_id])
                label = int(class_ids[roi_id])
                box = rois[roi_id, :]

                if score < self.threshold:
                    break

                image_result = pd.Series([label, float(score), box.tolist()], index=columns)

                results = results.append(image_result, ignore_index = True)
                


        results['x_pred'], results['y_pred'], results['w_pred'], results['h_pred'] = [i[0] for i in results['bbox']], [i[1] for i in results['bbox']], [i[2] for i in results['bbox']], [i[3] for i in results['bbox']]

        results['box_width'] = results['w_pred']/612*80
        results['box_height'] = results['h_pred']/768*80

        results['center_x'] = results['x_pred']/612*80 + results['box_width']/2 - 40
        results['center_y'] = 40 - results['y_pred']/768*80 - results['box_height']/2
        

        pred_corners = np.array([results.center_x + results.box_width/2, results.center_x + results.box_width/2,\
        results.center_x - results.box_width/2, results.center_x - results.box_width/2,\
        results.center_y + results.box_height/2, results.center_y - results.box_height/2,\
        results.center_y + results.box_height/2, results.center_y - results.box_height/2]).T

        pred_boxes = torch.as_tensor(pred_corners).view(-1, 2, 4)

        return [pred_boxes]

    def get_binary_road_map(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a cuda tensor with size [batch_size, 800, 800] 
        self.RoadMap_model.eval()
        samples = samples.view(1, 18, 256, 306).to(self.device)
        pred = self.RoadMap_model(samples)
        output = pred > 0.5
        # print(pred.shape)
        return output
