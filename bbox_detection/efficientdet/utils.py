import numpy as np
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import math
import cv2
import random
import pandas as pd



def convert_map_to_lane_map(ego_map, binary_lane):
    mask = (ego_map[0, :, :] == ego_map[1, :, :]) * (ego_map[1, :, :] == ego_map[2, :, :]) + (
                ego_map[0, :, :] == 250 / 255)

    if binary_lane:
        return (~ mask)
    return ego_map * (~ mask.view(1, ego_map.shape[1], ego_map.shape[2]))


def convert_map_to_road_map(ego_map):
    mask = (ego_map[0, :, :] == 1) * (ego_map[1, :, :] == 1) * (ego_map[2, :, :] == 1)

    return (~mask)


def collate_fn_dl(batch):
    return tuple(zip(*batch))


def draw_box(ax, corners, color):
    point_squence = torch.stack([corners[:, 0], corners[:, 1], corners[:, 3], corners[:, 2], corners[:, 0]])

    # the corners are in meter and time 10 will convert them in pixels
    # Add 400, since the center of the image is at pixel (400, 400)
    # The negative sign is because the y axis is reversed for matplotlib
    ax.plot(point_squence.T[0] * 10 + 400, -point_squence.T[1] * 10 + 400, color=color)


class BBoxTransform(nn.Module):
    def forward(self, anchors, regression):
        """
        decode_box_outputs adapted from https://github.com/google/automl/blob/master/efficientdet/anchors.py

        Args:
            anchors: [batchsize, boxes, (y1, x1, y2, x2)]
            regression: [batchsize, boxes, (dy, dx, dh, dw)]

        Returns:

        """
        # anchor center_y
        y_centers_a = (anchors[..., 0] + anchors[..., 2]) / 2
        # anchor center_x
        x_centers_a = (anchors[..., 1] + anchors[..., 3]) / 2
        # anchor height
        ha = anchors[..., 2] - anchors[..., 0]
        # anchor width
        wa = anchors[..., 3] - anchors[..., 1]

        w = regression[..., 3].exp() * wa
        h = regression[..., 2].exp() * ha

        y_centers = regression[..., 0] * ha + y_centers_a
        x_centers = regression[..., 1] * wa + x_centers_a

        ymin = y_centers - h / 2.
        xmin = x_centers - w / 2.
        ymax = y_centers + h / 2.
        xmax = x_centers + w / 2.

        return torch.stack([xmin, ymin, xmax, ymax], dim=2)


class ClipBoxes(nn.Module):

    def __init__(self):
        super(ClipBoxes, self).__init__()

    def forward(self, boxes, img):
        batch_size, num_channels, height, width = img.shape

        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
        boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

        boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width - 1)
        boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height - 1)

        return boxes


class Anchors(nn.Module):
    """
    adapted and modified from https://github.com/google/automl/blob/master/efficientdet/anchors.py by Zylo117
    """

    def __init__(self, anchor_scale=4., pyramid_levels=None, **kwargs):
        super().__init__()
        self.anchor_scale = anchor_scale

        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]

        self.strides = kwargs.get('strides', [2 ** x for x in self.pyramid_levels])
        self.scales = np.array(kwargs.get('scales', [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]))
        self.ratios = kwargs.get('ratios', [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)])

        self.last_anchors = {}
        self.last_shape = None

    def forward(self, image, dtype=torch.float32):
        """Generates multiscale anchor boxes.

        Args:
          image_size: integer number of input image size. The input image has the
            same dimension for width and height. The image_size should be divided by
            the largest feature stride 2^max_level.
          anchor_scale: float number representing the scale of size of the base
            anchor to the feature stride 2^level.
          anchor_configs: a dictionary with keys as the levels of anchors and
            values as a list of anchor configuration.

        Returns:
          anchor_boxes: a numpy array with shape [N, 4], which stacks anchors on all
            feature levels.
        Raises:
          ValueError: input size must be the multiple of largest feature stride.
        """
        image_shape = image.shape[2:]
        # print('----------------------------Anchors---------------------------------')
        # print('image shape selected, iamge original shape: {}, {}'.format(image_shape, image.shape))

        if image_shape == self.last_shape and image.device in self.last_anchors:
            return self.last_anchors[image.device]

        if self.last_shape is None or self.last_shape != image_shape:
            self.last_shape = image_shape

        if dtype == torch.float16:
            dtype = np.float16
        else:
            dtype = np.float32

        boxes_all = []
        for stride in self.strides:
            boxes_level = []
            for scale, ratio in itertools.product(self.scales, self.ratios):
                if image_shape[1] % stride != 0:
                    raise ValueError('input size must be divided by the stride.')
                base_anchor_size = self.anchor_scale * stride * scale
                anchor_size_x_2 = base_anchor_size * ratio[0] / 2.0
                anchor_size_y_2 = base_anchor_size * ratio[1] / 2.0

                x = np.arange(stride / 2, image_shape[1], stride)
                y = np.arange(stride / 2, image_shape[0], stride)
                xv, yv = np.meshgrid(x, y)
                xv = xv.reshape(-1)
                yv = yv.reshape(-1)

                # y1,x1,y2,x2
                boxes = np.vstack((yv - anchor_size_y_2, xv - anchor_size_x_2,
                                   yv + anchor_size_y_2, xv + anchor_size_x_2))
                boxes = np.swapaxes(boxes, 0, 1)
                boxes_level.append(np.expand_dims(boxes, axis=1))

            boxes_level = np.concatenate(boxes_level, axis=1)
            boxes_all.append(boxes_level.reshape([-1, 4]))

        anchor_boxes = np.vstack(boxes_all)

        anchor_boxes = torch.from_numpy(anchor_boxes.astype(dtype)).to(image.device)
        anchor_boxes = anchor_boxes.unsqueeze(0)

        # save it for later use to reduce overhead
        self.last_anchors[image.device] = anchor_boxes
        return anchor_boxes
    
    
    
class BEV:
    
    def __init__(self, image):
    
        self.image = image
        self.angle = [-30, -90, -150, 30, 90, 150]
        self.dst_h,self.dst_w = 400,560
        
    def bev_transform(self, x1=271, x2=289, crop_h = 140, dst_h=400, dst_w = 560):
        '''
        dst_h: destination image height
        dst_w: destination image height
        crop_h: crop height for ROI
        x1,x2: destination image correspongding points
        '''
        image = self.image
        
        H,W = image.shape[:2]

        #source image points
        src = np.float32([[0, H], [W, H], [0, 0], [W, 0]]) 
        #corresponding points in destination image
        dst = np.float32([[x1, self.dst_h], [x2, dst_h], [0, 0], [dst_w, 0]])
        M = cv2.getPerspectiveTransform(src, dst) # The transformation matrix

        image = image[crop_h:H, 0:W] # Apply np slicing for ROI crop

        self.warped = cv2.warpPerspective(image, M, (dst_w, dst_h)) # Image warping: transform to bird eye view
        img = self.warped
        n_h,n_w = img.shape[:2]
        mask = np.zeros(img.shape[:2], dtype=np.uint8)       
        points = np.array([[[x1+10, n_h], [x2-10, n_h], 
                            [515,0],[55, 0]]]) #make 60 degree angle crop


        cv2.drawContours(mask, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)
        self.warped_img = cv2.bitwise_and(img,img,mask = mask)
        
        return self.warped_img
    
    def whole_view(self, c_x=465, c_y = 465):
        
        warped_img = self.warped_img
        h,w = warped_img.shape[:2]
        self.whole_img = np.zeros((c_y*2,c_x*2,3))
        self.whole_img[c_y-h:c_x,c_y-w//2:c_y+w//2] = warped_img
        
        return self.whole_img
    
    def rotateImage(self, angle):
        image = self.whole_img
        dst_image = image.copy()
        (h, w) = image.shape[:2]
        (c_x, c_y) = (w // 2, h // 2)

        transl = np.array((2, 3))

        rotation_matrix = cv2.getRotationMatrix2D((c_x, c_y), angle, 1.0 )
        img_rotation = cv2.warpAffine(image, rotation_matrix, (w,h)) 

        return img_rotation


    
    

    
    
def bev_overview(image_list):
    for i,img in enumerate(image_list):
        bev = BEV(img)
        warped_img = bev.bev_transform() # transform single image to bev
        whole_img = bev.whole_view() # put bev segment into whole view
        rotated_img = bev.rotateImage(bev.angle[i]) # rotate by corresponding camera angle
        if i == 0:
            bev_img = rotated_img
        else:
            bev_img += rotated_img
    m,n = 400,400
    h,w = bev_img.shape[:2]
    cut_h,cut_w = (h-2*m)//2,(w-2*n)//2
    bev_img = bev_img[cut_h:cut_h+2*m,cut_w:cut_w+2*n]
    return bev_img

    
    
    

