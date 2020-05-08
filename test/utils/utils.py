
import os

import cv2
import numpy as np
import torch
from glob import glob
from torch import nn
from torchvision.ops import nms
from typing import Union
import uuid
import torchvision

from utils.sync_batchnorm import SynchronizedBatchNorm2d

from efficientdet.utils import BEV, bev_overview

def aspectaware_resize_padding(image, width, height, interpolation=None, means=None):

    old_h, old_w, c = image.shape
    if old_w > old_h:
        new_w = width
        new_h = int(width / old_w * old_h)
    else:
        new_w = int(height / old_h * old_w)
        new_h = height

    canvas = np.zeros((height, height, c), np.float32)
    if means is not None:
        canvas[...] = means

    if new_w != old_w or new_h != old_h:
        if interpolation is None:
            image = cv2.resize(image, (new_w, new_h))
        else:
            image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    padding_h = height - new_h
    padding_w = width - new_w

    if c > 1:
        canvas[:new_h, :new_w] = image
    else:
        if len(image.shape) == 2:
            canvas[:new_h, :new_w, 0] = image
        else:
            canvas[:new_h, :new_w] = image

    return [canvas, new_w, new_h, old_w, old_h, padding_w, padding_h]


def preprocess_test(sample, mode, max_size=512, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)):
    # Load original images in a list
    ori_imgs = []
    sample = sample.permute(0, 1, 3, 4, 2).numpy()

    image_front = []
    image_back = []
    images = []

    for i, image in enumerate(sample[0]):
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
    
    if mode == 'bev':
        image_set = [images[i].numpy().transpose(1, 2, 0) for i in range(len(images))]
        sample = bev_overview(image_set)
    else:
        sample = np.concatenate((image_back, image_front), axis=1)
    
    sample_lst = aspectaware_resize_padding(sample[..., ::-1], max_size, max_size, means=None)

    framed_img = sample_lst[0]

    framed_meta = sample_lst[1:]

    return sample, framed_img, framed_meta


def postprocess(x, anchors, regression, classification, regressBoxes, clipBoxes, threshold, iou_threshold):
    transformed_anchors = regressBoxes(anchors, regression)
    transformed_anchors = clipBoxes(transformed_anchors, x)
    scores = torch.max(classification, dim=2, keepdim=True)[0]
    scores_over_thresh = (scores > threshold)[:, :, 0]
    out = []

    for i in range(x.shape[0]):
        if scores_over_thresh.sum() == 0:
            out.append({
                'rois': np.array(()),
                'class_ids': np.array(()),
                'scores': np.array(()),
            })

        classification_per = classification[i, scores_over_thresh[i, :], ...].permute(1, 0)
        transformed_anchors_per = transformed_anchors[i, scores_over_thresh[i, :], ...]
        scores_per = scores[i, scores_over_thresh[i, :], ...]
        anchors_nms_idx = nms(transformed_anchors_per, scores_per[:, 0], iou_threshold=iou_threshold)

        if anchors_nms_idx.shape[0] != 0:
            scores_, classes_ = classification_per[:, anchors_nms_idx].max(dim=0)
            boxes_ = transformed_anchors_per[anchors_nms_idx, :]

            out.append({
                'rois': boxes_.cpu().numpy(),
                'class_ids': classes_.cpu().numpy(),
                'scores': scores_.cpu().numpy(),
            })
        else:
            out.append({
                'rois': np.array(()),
                'class_ids': np.array(()),
                'scores': np.array(()),
            })

    return out



def invert_affine(metas: Union[float, list, tuple], preds):
    for i in range(len(preds)):
        if len(preds[i]['rois']) == 0:
            continue
        else:
            if metas is float:
                preds[i]['rois'][:, [0, 2]] = preds[i]['rois'][:, [0, 2]] / metas
                preds[i]['rois'][:, [1, 3]] = preds[i]['rois'][:, [1, 3]] / metas
            else:
                new_w, new_h, old_w, old_h, padding_w, padding_h = metas[i]
                preds[i]['rois'][:, [0, 2]] = preds[i]['rois'][:, [0, 2]] / (new_w / old_w)
                preds[i]['rois'][:, [1, 3]] = preds[i]['rois'][:, [1, 3]] / (new_h / old_h)
    return preds




