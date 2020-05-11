# Author: Zylo117

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

image_names = [
    'CAM_FRONT_LEFT.jpeg',
    'CAM_FRONT.jpeg',
    'CAM_FRONT_RIGHT.jpeg',
    'CAM_BACK_LEFT.jpeg',
    'CAM_BACK.jpeg',
    'CAM_BACK_RIGHT.jpeg',
    ]
NUM_SAMPLE_PER_SCENE = 126

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

    return canvas, new_w, new_h, old_w, old_h, padding_w, padding_h,


def preprocess(*image_path, max_size=512, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)):
    ori_imgs = [cv2.imread(img_path) for img_path in image_path]
    normalized_imgs = [(img / 255 - mean) / std for img in ori_imgs]
    imgs_meta = [aspectaware_resize_padding(img[..., ::-1], max_size, max_size,
                                            means=None) for img in normalized_imgs]
    framed_imgs = [img_meta[0] for img_meta in imgs_meta]
    framed_metas = [img_meta[1:] for img_meta in imgs_meta]

    return ori_imgs, framed_imgs, framed_metas


def preprocess_dl(folder_path, val_index, mode = 'bev', max_size=512, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)):
    # Load original images in a list
    ori_imgs = []
    for scene_index in val_index:
        scene_path = os.path.join(folder_path, 'scene_{}'.format(scene_index))
        for sample_index in range(NUM_SAMPLE_PER_SCENE):
            sample_path = os.path.join(scene_path, 'sample_{}'.format(sample_index))

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

            if mode == 'bev':
                ori_imgs.append(bev_img.astype(np.float32)*255)
            else:
                ori_imgs.append(image_cat_2)


    #     ori_imgs = [cv2.imread(img_path) for img_path in image_path]

    # Normalize images
    normalized_imgs = [(img / 255 - mean) / std for img in ori_imgs]

    # Each image turn into canvas, new_w, new_h, old_w, old_h, padding_w, padding_h
    imgs_meta = [aspectaware_resize_padding(img[..., ::-1], max_size, max_size,
                                            means=None) for img in normalized_imgs]

    # canvas, resized padded images
    framed_imgs = [img_meta[0] for img_meta in imgs_meta]
    # metas
    framed_metas = [img_meta[1:] for img_meta in imgs_meta]

    return ori_imgs, framed_imgs, framed_metas



def preprocess_video(*frame_from_video, max_size=512, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)):
    ori_imgs = frame_from_video
    normalized_imgs = [(img / 255 - mean) / std for img in ori_imgs]
    imgs_meta = [aspectaware_resize_padding(img[..., ::-1], max_size, max_size,
                                            means=None) for img in normalized_imgs]
    framed_imgs = [img_meta[0] for img_meta in imgs_meta]
    framed_metas = [img_meta[1:] for img_meta in imgs_meta]

    return ori_imgs, framed_imgs, framed_metas

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


def display(preds, imgs, obj_list, imshow=True, imwrite=False):
    for i in range(len(imgs)):
        if len(preds[i]['rois']) == 0:
            continue

        for j in range(len(preds[i]['rois'])):
            (x1, y1, x2, y2) = preds[i]['rois'][j].astype(np.int)
            cv2.rectangle(imgs[i], (x1, y1), (x2, y2), (255, 255, 0), 2)
            obj = obj_list[preds[i]['class_ids'][j]]
            score = float(preds[i]['scores'][j])

            cv2.putText(imgs[i], '{}, {:.3f}'.format(obj, score),
                        (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 0), 1)
        if imshow:
            cv2.imshow('img', imgs[i])
            cv2.waitKey(0)

        if imwrite:
            os.makedirs('test/', exist_ok=True)
            cv2.imwrite(f'test/{uuid.uuid4().hex}.jpg', imgs[i])


def replace_w_sync_bn(m):
    for var_name in dir(m):
        target_attr = getattr(m, var_name)
        if type(target_attr) == torch.nn.BatchNorm2d:
            num_features = target_attr.num_features
            eps = target_attr.eps
            momentum = target_attr.momentum
            affine = target_attr.affine

            # get parameters
            running_mean = target_attr.running_mean
            running_var = target_attr.running_var
            if affine:
                weight = target_attr.weight
                bias = target_attr.bias

            setattr(m, var_name,
                    SynchronizedBatchNorm2d(num_features, eps, momentum, affine))

            target_attr = getattr(m, var_name)
            # set parameters
            target_attr.running_mean = running_mean
            target_attr.running_var = running_var
            if affine:
                target_attr.weight = weight
                target_attr.bias = bias

    for var_name, children in m.named_children():
        replace_w_sync_bn(children)


class CustomDataParallel(nn.DataParallel):
    """
    force splitting data to all gpus instead of sending all data to cuda:0 and then moving around.
    """

    def __init__(self, module, num_gpus):
        super().__init__(module)
        self.num_gpus = num_gpus

    def scatter(self, inputs, kwargs, device_ids):
        # More like scatter and data prep at the same time. The point is we prep the data in such a way
        # that no scatter is necessary, and there's no need to shuffle stuff around different GPUs.
        devices = ['cuda:' + str(x) for x in range(self.num_gpus)]
        splits = inputs[0].shape[0] // self.num_gpus

        return [(inputs[0][splits * device_idx: splits * (device_idx + 1)].to(f'cuda:{device_idx}', non_blocking=True),
                 inputs[1][splits * device_idx: splits * (device_idx + 1)].to(f'cuda:{device_idx}', non_blocking=True))
                for device_idx in range(len(devices))], \
               [kwargs] * len(devices)


def get_last_weights(weights_path):
    weights_path = glob(weights_path + f'/*.pth')
    weights_path = sorted(weights_path,
                          key=lambda x: int(x.rsplit('_')[-1].rsplit('.')[0]),
                          reverse=True)[0]
    print(f'using weights {weights_path}')
    return weights_path


def init_weights(model):
    for name, module in model.named_modules():
        is_conv_layer = isinstance(module, nn.Conv2d)

        if is_conv_layer:
            nn.init.kaiming_uniform_(module.weight.data)

            if module.bias is not None:
                module.bias.data.zero_()

def loss_writer(saved_path, cls_loss_ls, reg_loss_ls, epoch_loss, current_lr, step, epoch, val = False):
    with open(os.path.join(saved_path, 'loss_log.txt'), 'a+') as f:
        if val == False:
            avg_cls_loss = np.mean(cls_loss_ls)
            avg_reg_loss = np.mean(reg_loss_ls)
            avg_loss = np.mean(epoch_loss)
            line = 'Step {:d}: Current Classification Loss:{:4f}; Current Regression Loss:{:4f}; Current Loss:{:4f}; \
             Average Classification Loss:{:4f}; Average Regression Loss:{:4f}; Average Loss:{:4f}; Current Learning Rate:{:4f}.'\
                .format(step, cls_loss_ls[-1], reg_loss_ls[-1], epoch_loss[-1], avg_cls_loss, avg_reg_loss, avg_loss, current_lr)
            print(line)
            f.write(line + '\n')
        else:
            line = '--------------------------------------------After Epoch {}------------------------------------------------\n'.format(epoch) +\
                   'Step {:d}: Validation Classification Loss:{:4f}; Regression Loss:{:4f}; Total Loss:{:4f}\n'.format(step, cls_loss_ls[0], reg_loss_ls[0], epoch_loss[0]) +\
                   '----------------------------------------------------------------------------------------------------------'
            print(line)
            f.write(line + '\n')

def save_model(model, best_model, current_model, best_loss, current_loss, saved_model_path, step, compound_coef, mode, val = False):
    save_dir = saved_model_path

    print('Save current model ...')
    if current_model is not None:
        model_path = os.path.join(save_dir, current_model)
        os.remove(model_path)

    if val == True:
        current_model = f'{mode}_efficientdet-d{compound_coef}_{step}_val.pth'
    else:
        current_model = f'{mode}_efficientdet-d{compound_coef}_{step}.pth'

    model_path = os.path.join(save_dir, current_model)
    torch.save(model.module.model.state_dict(), model_path)


    print('Save best model ...')
    if best_model is not None:
        if current_loss < best_loss:
            old_dir = best_model
            os.remove(os.path.join(save_dir, old_dir))
    if val == True:
        best_model = f'best-{mode}_efficientdet-d{compound_coef}_{step}_val.pth'
    else:
        best_model = f'best-{mode}_efficientdet-d{compound_coef}_{step}.pth'


    best_loss = current_loss
    model_path = os.path.join(save_dir, best_model)
    torch.save(model.module.model.state_dict(), model_path)



    return best_model, best_loss, current_model


def save_with_epoch(model, save_dir, run_name, best_model = False):
    best_prefix = 'best-' if best_model else ''
    model_dir = os.path.join(save_dir, best_prefix + run_name)
    torch.save(model, model_dir)