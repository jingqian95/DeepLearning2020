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

from efficientdet.utils import convert_map_to_lane_map, convert_map_to_road_map, BEV, bev_overview

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


class LabeledDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, annotation_file, scene_index, transform, extra_info=True):
        """
        Args:
            image_folder (string): the location of the image folder
            annotation_file (string): the location of the annotations
            scene_index (list): a list of scene indices for the unlabeled data
            transform (Transform): The function to process the image
            extra_info (Boolean): whether you want the extra information
        """

        self.image_folder = image_folder
        self.annotation_dataframe = pd.read_csv(annotation_file)
        self.scene_index = scene_index
        self.transform = transform
        self.extra_info = extra_info

    def __len__(self):
        return self.scene_index.size * NUM_SAMPLE_PER_SCENE

    def __getitem__(self, index):
        scene_id = self.scene_index[index // NUM_SAMPLE_PER_SCENE]
        sample_id = index % NUM_SAMPLE_PER_SCENE
        sample_path = os.path.join(self.image_folder, f'scene_{scene_id}', f'sample_{sample_id}')

        images = []
        image_cat = None
        for image_name in image_names:
            image_path = os.path.join(sample_path, image_name)
            image = Image.open(image_path)
            image = self.transform(image)
            if image_cat == None:
                image_cat = image
            else:
                image_cat = torch.cat((image_cat, image), 1)
            images.append(image)

        image_tensor = torch.stack(images)

        data_entries = self.annotation_dataframe[
            (self.annotation_dataframe['scene'] == scene_id) & (self.annotation_dataframe['sample'] == sample_id)]
        corners = data_entries[['fl_x', 'fr_x', 'bl_x', 'br_x', 'fl_y', 'fr_y', 'bl_y', 'br_y']].to_numpy()
        categories = data_entries.category_id.to_numpy()

        ego_path = os.path.join(sample_path, 'ego.png')
        ego_image = Image.open(ego_path)
        ego_image = torchvision.transforms.functional.to_tensor(ego_image)
        road_image = convert_map_to_road_map(ego_image)

        target = {}
        target['bounding_box'] = torch.as_tensor(corners).view(-1, 2, 4)
        target['category'] = torch.as_tensor(categories)

        if self.extra_info:
            actions = data_entries.action_id.to_numpy()
            # You can change the binary_lane to False to get a lane with
            lane_image = convert_map_to_lane_map(ego_image, binary_lane=True)

            extra = {}
            extra['action'] = torch.as_tensor(actions)
            extra['ego_image'] = ego_image
            extra['lane_image'] = lane_image

            return image_cat, image_tensor, target, road_image, extra

        else:
            return image_cat, image_tensor, target, road_image

class LabeledDataset_dl(torch.utils.data.Dataset):
    def __init__(self, image_folder, annotation_file, scene_index, transform, extra_info=True):
        """
        Args:
            image_folder (string): the location of the image folder
            annotation_file (string): the location of the annotations
            scene_index (list): a list of scene indices for the unlabeled data
            transform (Transform): The function to process the image
            extra_info (Boolean): whether you want the extra information
        """

        self.image_folder = image_folder
        self.annotation_dataframe = pd.read_csv(annotation_file)
        self.scene_index = scene_index
        self.transform = transform
        self.extra_info = extra_info


        self.load_classes()

    def load_classes(self):

        # load class names (name -> label)
        categories = ['other_vehicle', 'bicycle', 'car', 'pedestrian', 'truck', 'bus', 'motorcycle', 'emergency_vehicle', 'animal']


        self.classes = {}
        self.coco_labels = {}
        self.coco_labels_inverse = {}
        for c in range(9):
            self.coco_labels[len(self.classes)] = c
            self.coco_labels_inverse[c] = len(self.classes)
            self.classes[categories[c]] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def __len__(self):
        return self.scene_index.size * NUM_SAMPLE_PER_SCENE

    def __getitem__(self, index):

        bev_img, img = self.load_image(index)
        # print('image done')
        annot, bev_annot = self.load_annotations(index)
        # print('annotation done')
        roadimage = self.load_roadimage(index)
        sample = {'img': img, 'annot': annot, 'bev_img': bev_img, 'bev_annot': bev_annot, 'roadimage': roadimage}

        if self.transform:
            sample = self.transform(sample)


        return sample

    def load_image(self, image_index):
        scene_id = self.scene_index[image_index // NUM_SAMPLE_PER_SCENE]
        sample_id = image_index % NUM_SAMPLE_PER_SCENE
        sample_path = os.path.join(self.image_folder, f'scene_{scene_id}', f'sample_{sample_id}')


        image_front = []
        image_back = []
        images = []

        for i in range(6):
            image_path = os.path.join(sample_path, image_names[i])
            image = cv2.imread(image_path)

            transform_dl = torchvision.transforms.ToTensor()
            images.append(transform_dl(image))

            if i <= 2:
                if len(image_front) < 1:
                    image_front = image
                else:
                    image_front = np.concatenate((image_front, image), axis=0)
            else:
                if len(image_back) < 1:
                    image_back = image
                else:
                    image_back = np.concatenate((image_back, image), axis=0)



        image_cat_2 = np.concatenate((image_back, image_front), axis=1)
        
        image_set = [images[i].numpy().transpose(1, 2, 0) for i in range(len(images))]
        
        bev_img = bev_overview(image_set)

        image_tensor = torch.stack(images)


        bev_img = cv2.cvtColor(bev_img.astype(np.float32)*255, cv2.COLOR_BGR2RGB)
        image_cat_2 = cv2.cvtColor(image_cat_2, cv2.COLOR_BGR2RGB)

        return bev_img.astype(np.float32)/255, image_cat_2.astype(np.float32) / 255.



    def load_annotations(self, image_index):
        # get ground truth annotations
        scene_id = self.scene_index[image_index // NUM_SAMPLE_PER_SCENE]
        sample_id = image_index % NUM_SAMPLE_PER_SCENE
        data_entries = self.annotation_dataframe[
            (self.annotation_dataframe['scene'] == scene_id) & (self.annotation_dataframe['sample'] == sample_id)]
        annotations = np.zeros((0, 5))
        # parse annotations
        for idx, a in data_entries.iterrows():

            annotation = np.zeros((1, 5))
            bev_annotation = np.zeros((1, 5))
            
            annotation[0, :4] = a['scaled_x'], a['scaled_y'], a['scaled_box_width'], a['scaled_box_height']
            bev_annotation[0, :4] = a['bev_x'], a['bev_y'], a['bev_box_width'], a['bev_box_height']
            annotation[0, 4] = self.coco_label_to_label(a['category_id'])
            annotations = np.append(annotations, annotation, axis=0)
            bev_annotation[0, 4] = self.coco_label_to_label(a['category_id'])
            bev_annotation = np.append(bev_annotation, bev_annotation, axis=0)


        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]
        bev_annotation[:, 2] = bev_annotation[:, 0] + bev_annotation[:, 2]
        bev_annotation[:, 3] = bev_annotation[:, 1] + bev_annotation[:, 3]


        return annotations, bev_annotation

    def load_roadimage(self, image_index):
        # get road image
        scene_id = self.scene_index[image_index // NUM_SAMPLE_PER_SCENE]
        sample_id = image_index % NUM_SAMPLE_PER_SCENE
        sample_path = os.path.join(self.image_folder, f'scene_{scene_id}', f'sample_{sample_id}')

        ego_path = os.path.join(sample_path, 'ego.png')
        ego_image = Image.open(ego_path)
        ego_image = torchvision.transforms.functional.to_tensor(ego_image)
        road_image = convert_map_to_road_map(ego_image)

        return road_image

    def coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]

    def label_to_coco_label(self, label):
        return self.coco_labels[label]


def collater(data):
    imgs = [s['img'] for s in data]
    annot = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]
    bev_imgs = [s['bev_img'] for s in data]
    bev_annot = [s['bev_annot'] for s in data]
    bev_scales = [s['bev_scale'] for s in data]
    roadimage = [s['roadimage'] for s in data]

    imgs = torch.from_numpy(np.stack(imgs, axis=0))
    bev_imgs = torch.from_numpy(np.stack(bev_imgs, axis=0))

    max_num_annot = max(annot.shape[0] for annot in annot)
    bev_max_num_annot = max(bev_annot.shape[0] for bev_annot in bev_annot)

    if max_num_annot > 0:

        annot_padded = torch.ones((len(annot), max_num_annot, 5)) * -1

        if max_num_annot > 0:
            for idx, annot in enumerate(annot):
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annot), 1, 5)) * -1
        
    if bev_max_num_annot > 0:

        bev_annot_padded = torch.ones((len(bev_annot), bev_max_num_annot, 5)) * -1

        if bev_max_num_annot > 0:
            for bev_idx, bev_annot in enumerate(bev_annot):
                if bev_annot.shape[0] > 0:
                    bev_annot_padded[bev_idx, :bev_annot.shape[0], :] = bev_annot
    else:
        bev_annot_padded = torch.ones((len(bev_annot), 1, 5)) * -1
        

    imgs = imgs.permute(0, 3, 1, 2)
    bev_imgs = bev_imgs.permute(0, 3, 1, 2)

    return {'img': imgs, 'annot': annot_padded, 'scale': scales,\
            'bev_img': bev_imgs, 'bev_annot': bev_annot_padded, 'bev_scale': bev_scales,\
            'roadimage': roadimage}

#
class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, img_size=512):
        self.img_size = img_size

    def __call__(self, sample):
        image, annot, bev, bev_annot, roadimage = sample['img'], sample['annot'], sample['bev_img'],\
                sample['bev_annot'],sample['roadimage']
        height, width, _ = image.shape
        b_height, b_width, _ = bev.shape
        
        if height > width:
            scale = self.img_size / height
            resized_height = self.img_size
            resized_width = int(width * scale)
        else:
            scale = self.img_size / width
            resized_height = int(height * scale)
            resized_width = self.img_size
            
        if b_height > b_width:
            b_scale = self.img_size / b_height
            b_resized_height = self.img_size
            b_resized_width = int(b_width * b_scale)
        else:
            b_scale = self.img_size / b_width
            b_resized_height = int(b_height * b_scale)
            b_resized_width = self.img_size

        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        bev = cv2.resize(bev, (b_resized_width, b_resized_height), interpolation=cv2.INTER_LINEAR)

        new_image = np.zeros((self.img_size, self.img_size, 3))
        new_image[0:resized_height, 0:resized_width] = image
        
        b_new_image = np.zeros((self.img_size, self.img_size, 3))
        b_new_image[0:b_resized_height, 0:b_resized_width] = bev

        annot[:, :4] *= scale
        bev_annot[:, :4] *= b_scale

        return {'img': torch.from_numpy(new_image).to(torch.float32), 'annot': torch.from_numpy(annot),\
                'bev_img': torch.from_numpy(b_new_image).to(torch.float32), 'bev_annot': torch.from_numpy(bev_annot),\
                'scale': scale, 'bev_scale': b_scale, 'roadimage': roadimage}
    
    


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):
        if np.random.rand() < flip_x:
            image, annot, bev, bev_annot, roadimage = sample['img'], sample['annot'], sample['bev_img'],\
                sample['bev_annot'],sample['roadimage']
            image = image[:, ::-1, :]
            bev = bev[:, ::-1, :]

            rows, cols, channels = image.shape
            b_rows, b_cols, b_channels = bev.shape
            
            x1 = annot[:, 0].copy()
            x2 = annot[:, 2].copy()
            
            b_x1 = bev_annot[:, 0].copy()
            b_x2 = bev_annot[:, 2].copy()

            x_tmp = x1.copy()
            b_x_tmp = b_x1.copy()

            annot[:, 0] = cols - x2
            annot[:, 2] = cols - x_tmp
            
            bev_annot[:, 0] = b_cols - b_x2
            bev_annot[:, 2] = b_cols - b_x_tmp


            sample = {'img': image, 'annot': annot, 'bev_img': bev, 'bev_annot': bev_annot, 'roadimage': roadimage}


        return sample



class Normalizer(object):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])

    def __call__(self, sample):
        image, annot, bev, bev_annot, roadimage = sample['img'], sample['annot'], sample['bev_img'],\
                sample['bev_annot'],sample['roadimage']
        
        return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'annot': annot, \
                'bev_img': ((bev.astype(np.float32) - self.mean) / self.std), 'bev_annot': bev_annot,\
                'roadimage': roadimage}
    
    
    
    
    
    
    
    
    

