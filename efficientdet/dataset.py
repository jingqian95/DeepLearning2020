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

            # transform_dl = torchvision.transforms.ToTensor()
            # image = transform_dl(image)

            # print(image_name)
            # print('before transform image shape {}'.format(image.shape))
            image = self.transform(image)
            # print('after transform image shape {}'.format(image.shape))
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

class LabeledDataset_coco(torch.utils.data.Dataset):
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

        # self.root_dir = root_dir
        # self.set_name = set
        # self.transform = transform

        # self.coco = COCO(os.path.join(self.root_dir, 'annotations', 'instances_' + self.set_name + '.json'))
        # self.image_ids = self.coco.getImgIds()

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

        img = self.load_image(index)
        # print('image done')
        annot = self.load_annotations(index)
        # print('annotation done')
        roadimage = self.load_roadimage(index)
        sample = {'img': img, 'annot': annot}

        if self.transform:
            sample = self.transform(sample)

        # print(sample['img'].shape, sample['annot'].shape)
        # print('----------------------------LabeledDataset_coco_output---------------------------------')
        # print('After LabeledDataset_coco annotations(106,0) shape: {}\nValue'.format(sample['annot'].shape))
        # print(sample['annot'][:5])

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
        image_cat = np.concatenate((image_front, image_back), axis=0)


        image_tensor = torch.stack(images)

        image_cat = cv2.cvtColor(image_cat, cv2.COLOR_BGR2RGB)
        image_cat_2 = cv2.cvtColor(image_cat_2, cv2.COLOR_BGR2RGB)
        print('image_size: {}'.format(image_cat_2.shape))

        return image_cat_2.astype(np.float32) / 255.



    def load_annotations(self, image_index):
        # get ground truth annotations
        scene_id = self.scene_index[image_index // NUM_SAMPLE_PER_SCENE]
        sample_id = image_index % NUM_SAMPLE_PER_SCENE
        data_entries = self.annotation_dataframe[
            (self.annotation_dataframe['scene'] == scene_id) & (self.annotation_dataframe['sample'] == sample_id)]
        # print(data_entries.shape)
        # print('----------------------------load_annotations---------------------------------')
        annotations = np.zeros((0, 5))
        # parse annotations
        for idx, a in data_entries.iterrows():

            annotation = np.zeros((1, 5))
            annotation[0, :4] = a['scaled_x'], a['scaled_y'], a['scaled_box_width'], a['scaled_box_height']
            # annotation[0, :4] = (a['center_x']+40)*768/80, (a['center_y']+40)*612/80, a['box_width']*768/80, a['box_height']*612/80
            annotation[0, 4] = self.coco_label_to_label(a['category_id'])
            annotations = np.append(annotations, annotation, axis=0)

        # print('----------------------------After input---------------------------------')
        # print('After input annotations(106,0) shape: {}\nValue'.format(annotations.shape))
        # print(annotations[:5])

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        # print('----------------------------After transform---------------------------------')
        # print('After transform annotations(106,0) shape: {}\nValue'.format(annotations.shape))
        # print(annotations[:5])

        return annotations

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
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]

    imgs = torch.from_numpy(np.stack(imgs, axis=0))

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    imgs = imgs.permute(0, 3, 1, 2)

    return {'img': imgs, 'annot': annot_padded, 'scale': scales}

#
class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, img_size=512):
        self.img_size = img_size

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        height, width, _ = image.shape
        if height > width:
            scale = self.img_size / height
            resized_height = self.img_size
            resized_width = int(width * scale)
        else:
            scale = self.img_size / width
            resized_height = int(height * scale)
            resized_width = self.img_size

        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        new_image = np.zeros((self.img_size, self.img_size, 3))
        new_image[0:resized_height, 0:resized_width] = image

        annots[:, :4] *= scale
        # print('----------------------------Resizer---------------------------------')
        # print('After Resizer annotations(106,0) shape: {}\nValue'.format(torch.from_numpy(annots).shape))
        # print(torch.from_numpy(annots)[:5])

        return {'img': torch.from_numpy(new_image).to(torch.float32), 'annot': torch.from_numpy(annots), 'scale': scale}

# class Resizer(object):
#     """Convert ndarrays in sample to Tensors."""
#
#     def __init__(self, img_size=512):
#         self.img_size = img_size
#
#     def __call__(self, sample):
#         image = sample
#         height, width, _ = image.shape
#         if height > width:
#             scale = self.img_size / height
#             resized_height = self.img_size
#             resized_width = int(width * scale)
#         else:
#             scale = self.img_size / width
#             resized_height = int(height * scale)
#             resized_width = self.img_size
#
#         image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
#
#         new_image = np.zeros((self.img_size, self.img_size, 3))
#         new_image[0:resized_height, 0:resized_width] = image
#
#         annots[:, :4] *= scale
#
#         # print(image.shape)
#         return image



class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):
        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            # print('----------------------------Augmenter_True---------------------------------')
            # print('After Augmenter annotations(106,0) shape: {}\nValue'.format(annots.shape))
            # print(annots[:5])

            sample = {'img': image, 'annot': annots}
        # else:
            # print('----------------------------Augmenter_False---------------------------------')
            # print('After Augmenter annotations(106,0) shape: {}\nValue'.format(sample['annot'].shape))
            # print(sample['annot'][:5])

        return sample



# class Augmenter(object):
#     """Convert ndarrays in sample to Tensors."""
#
#     def __call__(self, sample, flip_x=0.5):
#         if np.random.rand() < flip_x:
#             image = sample
#             image = image[:, ::-1, :]
#
#             rows, cols, channels = image.shape
#
#             # x1 = annots[:, 0].copy()
#             # x2 = annots[:, 2].copy()
#             #
#             # x_tmp = x1.copy()
#             #
#             # annots[:, 0] = cols - x2
#             # annots[:, 2] = cols - x_tmp
#
#             sample = image
#
#         return sample

class Normalizer(object):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        # print('----------------------------Normalizer---------------------------------')
        # print('After Normalizer annotations(106,0) shape: {}\nValue'.format(annots.shape))
        # print(annots[:5])
        return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'annot': annots}


# class Normalizer(object):
#
#     def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
#         self.mean = np.array([[mean]])
#         self.std = np.array([[std]])
#
#     def __call__(self, sample):
#         image = sample
#
#         return (image - self.mean) / self.std
