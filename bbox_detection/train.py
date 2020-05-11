# original author: Zylo117
# adapted from https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch
# modified by muyangjin

import datetime
import os
import argparse
import traceback

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
import torch.nn.functional as F
import torch.backends.cudnn as cudnn


from efficientdet.dataset import LabeledDataset_dl, Resizer, Normalizer, Augmenter, collater

from backbone import EfficientDetBackbone
from tensorboardX import SummaryWriter
import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm
from efficientdet.loss import FocalLoss
from efficientdet.utils import collate_fn_dl
from utils.sync_batchnorm import patch_replication_callback
from utils.utils import replace_w_sync_bn, CustomDataParallel, get_last_weights, init_weights, loss_writer, save_model, save_with_epoch

from yacs.config import CfgNode as CN

class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)


def get_args():
    parser = argparse.ArgumentParser('Yet Another EfficientDet Pytorch: SOTA object detection network - Zylo117')
    parser.add_argument('-p', '--project', type=str, default='dl2020', help='project file that contains parameters')
    parser.add_argument('-c', '--compound_coef', type=int, default=0, help='coefficients of efficientdet')
    parser.add_argument('-n', '--num_workers', type=int, default=4, help='num_workers of dataloader')
    parser.add_argument('--batch_size', type=int, default=10, help='The number of images per batch among all devices')
    parser.add_argument('--head_only', type=bool, default=False,
                        help='whether finetunes only the regressor and the classifier, '
                             'useful in early stage convergence or small/easy dataset')
    parser.add_argument('--mode', type=str, default='bev', help = 'Options: bev, ori')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--optim', type=str, default='adamw', help='select optimizer for training, '
                                                                   'suggest using \'admaw\' until the'
                                                                   ' very final stage then switch to \'sgd\'')
    parser.add_argument('--alpha', type=float, default=0.25)
    parser.add_argument('--gamma', type=float, default=1.5)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--val_interval', type=int, default=1, help='Number of epoches between valing phases')
    parser.add_argument('--save_interval', type=int, default=100, help='Number of steps between saving')
    parser.add_argument('--es_min_delta', type=float, default=0.0,
                        help='Early stopping\'s parameter: minimum change loss to qualify as an improvement')
    parser.add_argument('--es_patience', type=int, default=0,
                        help='Early stopping\'s parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.')
    parser.add_argument('--data_path', type=str, default='datasets/', help='the root folder of dataset')
    parser.add_argument('--annotation', type=str, default='annotation_newfeat_3.csv', help='annotation csv file name')
    # parser.add_argument('--log_path', type=str, default='saved/')
    parser.add_argument('--load_weights', type=str, default=None,
                        help='whether to load weights from a checkpoint, set None to initialize, set \'last\' to load last checkpoint')
    parser.add_argument('--saved_path', type=str, default='saved/')
    parser.add_argument('--debug', type=bool, default=False, help='whether visualize the predicted boxes of trainging, '
                                                                  'the output images will be in test/')

    args = parser.parse_args()
    return args





def train(opt):
    params = Params(f'projects/{opt.project}.yml')

    if params.num_gpus == 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    else:
        torch.manual_seed(42)


    folder_name = '{}_{}_coef{}_{}/'.format(opt.project,  datetime.datetime.now().strftime("%m%d-%H%M%S"), opt.compound_coef, opt.mode)



    opt.saved_path = opt.saved_path + f'/{params.project_name}/' + folder_name
    opt.saved_model_path = opt.saved_path  + 'model/'
    # opt.log_path = opt.saved_path + f'/{params.project_name}/tensorboard/'
    os.makedirs(opt.saved_path, exist_ok=True)
    os.makedirs(opt.saved_model_path, exist_ok=True)

    #Write config
    with open(opt.saved_path + '/config.txt', 'w') as f:
        f.write(str(opt))

    #Initialize training parameters
    training_params = {'batch_size': opt.batch_size,
                       'shuffle': True,
                       'drop_last': True,
                       'collate_fn': collater,
                       'num_workers': opt.num_workers}

    val_params = {'batch_size': opt.batch_size,
                  'shuffle': False,
                  'drop_last': True,
                  'collate_fn': collater,
                  'num_workers': opt.num_workers}



    train_scene_index = np.arange(int(params.train_set.split(',')[0]), int(params.train_set.split(',')[1]) + 1)
    val_scene_index = np.arange(int(params.val_set.split(',')[0]), int(params.val_set.split(',')[1]) + 1)
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]

    # transform_dl = torchvision.transforms.ToTensor()

    transform = transforms.Compose([Normalizer(mean=params.mean, std=params.std),\
                                    Augmenter(),\
                                    Resizer(input_sizes[opt.compound_coef])])

    training_set = LabeledDataset_dl(image_folder = os.path.join(opt.data_path, params.project_name),
                                      annotation_file = os.path.join(opt.data_path, params.project_name, opt.annotation),
                                      scene_index = train_scene_index,
                                      transform = transform,
                                      extra_info = True
                                      )

    training_generator = DataLoader(training_set, **training_params)


    val_set = LabeledDataset_dl(image_folder = os.path.join(opt.data_path, params.project_name),
                                      annotation_file = os.path.join(opt.data_path, params.project_name, opt.annotation),
                                      scene_index = val_scene_index,
                                      transform = transform,
                                      extra_info = True
                                      )

    val_generator = DataLoader(val_set, **val_params)



    print('Training Object Detector...')

    # Initialize Model
    model = EfficientDetBackbone(num_classes=len(params.obj_list), compound_coef = opt.compound_coef,
                             ratios=eval(params.anchors_ratios), scales=eval(params.anchors_scales))



    # load last weights
    if opt.load_weights is not None:
        if opt.load_weights.endswith('.pth'):
            weights_path = opt.load_weights
        else:
            weights_path = get_last_weights(opt.saved_path)
        try:
            last_step = int(os.path.basename(weights_path).split('_')[-1].split('.')[0])
        except:
            last_step = 0

        try:
            ret = model.load_state_dict(torch.load(weights_path), strict=False)
        except RuntimeError as e:
            print(f'[Warning] Ignoring {e}')
            print(
                '[Warning] Don\'t panic if you see this, this might be because you load a pretrained weights with different number of classes. The rest of the weights should be loaded already.')

        print(f'[Info] loaded weights: {os.path.basename(weights_path)}, resuming checkpoint from step: {last_step}')
    else:
        last_step = 0
        print('[Info] initializing weights...')
        init_weights(model)




    # freeze backbone if train head_only
    if opt.head_only:
        def freeze_backbone(m):
            classname = m.__class__.__name__
            for ntl in ['EfficientNet', 'BiFPN']:
                if ntl in classname:
                    for param in m.parameters():
                        param.requires_grad = False

        model.apply(freeze_backbone)
        print('[Info] freezed backbone')



    # https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
    # apply sync_bn when using multiple gpu and batch_size per gpu is lower than 4
    #  useful when gpu memory is limited.
    # because when bn is disable, the training will be very unstable or slow to converge,
    # apply sync_bn can solve it,
    # by packing all mini-batch across all gpus as one batch and normalize, then send it back to all gpus.
    # but it would also slow down the training by a little bit.
    if params.num_gpus > 1 and opt.batch_size // params.num_gpus < 4:
        model.apply(replace_w_sync_bn)
        use_sync_bn = True
    else:
        use_sync_bn = False


    # Initiate Log writer
    # writer = SummaryWriter(opt.log_path + f'/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}/')


    # warp the model with loss function, to reduce the memory usage on gpu0 and speedup
    class ModelWithLoss(nn.Module):
        def __init__(self, model, debug=False):
            super().__init__()
            self.criterion = FocalLoss()
            self.model = model
            self.debug = debug

        def forward(self, imgs, annotations, obj_list=None):
            _, regression, classification, anchors = self.model(imgs)
            if self.debug:
                cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations,
                                                    imgs=imgs, obj_list=obj_list)
            else:
                cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations)
            return cls_loss, reg_loss, regression, classification, anchors


    model = ModelWithLoss(model, debug=False)

    # Put model into Cuda
    if params.num_gpus > 0:
        model = model.cuda()
        if params.num_gpus > 1:
            model = CustomDataParallel(model, params.num_gpus)
            if use_sync_bn:
                patch_replication_callback(model)

    # Initialize optimizer and criterion(obj_det done above in ModelWithLoss)

    if opt.optim == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), opt.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), opt.lr, momentum=0.9, nesterov=True)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)



    epoch = 0
    best_loss = None
    best_epoch = 0
    step = max(0, last_step)
    model.train()
    best_model = None
    current_model = None
    best_val_loss = None
    best_val_model = None
    current_val_model = None

    num_iter_per_epoch = len(training_generator)



    try:
        for epoch in range(opt.num_epochs):
            last_epoch = step // num_iter_per_epoch
            reg_loss_ls = []
            cls_loss_ls = []
            if epoch < last_epoch:
                continue

            epoch_loss = []

            progress_bar = tqdm(training_generator)

            for iter, data in enumerate(progress_bar):
                if iter < step - last_epoch * num_iter_per_epoch:
                    progress_bar.update()
                    continue

                try:
                    if opt.mode == 'bev':
                        imgs = data['bev_img']
                        annot = data['bev_annot']
                    else:
                        imgs = data['img']
                        annot = data['annot']

                    # sample_cat, sample, target, road_image, extra = data

                    if params.num_gpus == 1:
                        # if only one gpu, just send it to cuda:0
                        # elif multiple gpus, send it to multiple gpus in CustomDataParallel, not here
                        imgs = imgs.cuda()
                        annot = annot.cuda()

                    optimizer.zero_grad()
                    cls_loss, reg_loss, regression, classification, anchors = model(imgs, annot,\
                                                                                    obj_list=params.obj_list)
                    cls_loss = cls_loss.mean()
                    cls_loss_ls.append(float(cls_loss))
                    reg_loss = reg_loss.mean()
                    reg_loss_ls.append(float(reg_loss))

                    loss = cls_loss + reg_loss
                    if loss == 0 or not torch.isfinite(loss):
                        continue

                    loss.backward()

                    if best_loss is None:
                        best_loss = loss

                    # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    optimizer.step()

                    epoch_loss.append(float(loss))



                    # log learning_rate
                    current_lr = optimizer.param_groups[0]['lr']


                    progress_bar.set_description('Object Detection: Step: {}. Epoch: {}/{}. Iteration: {}/{}. Cls loss: {:.5f}. Reg loss: {:.5f}. Total loss: {:.5f}'.format(
                                step, epoch, opt.num_epochs, iter + 1, num_iter_per_epoch, cls_loss.item(),
                                reg_loss.item(), loss.item()))

                    if step % opt.save_interval == 0:
                        loss_writer(opt.saved_path, cls_loss_ls, reg_loss_ls, epoch_loss, current_lr, step, epoch)



                    step += 1

                    if step % opt.save_interval == 0 and step > 0:
                        # print('Save Model')
                        best_model, best_loss, current_model = \
                            save_model(model, best_model, current_model, best_loss, loss, opt.saved_model_path, step, opt.compound_coef, opt.mode)

                except Exception as e:
                    print('[Error]', traceback.format_exc())
                    print(e)
                    continue

            scheduler.step(np.mean(epoch_loss))

            if epoch % opt.val_interval == 0:

                    model.eval()
                    loss_regression_ls = []
                    loss_classification_ls = []
                    for iter, data in enumerate(val_generator):
                        with torch.no_grad():

                            if opt.mode == 'bev':
                                imgs = data['bev_img']
                                annot = data['bev_annot']
                            else:
                                imgs = data['img']
                                annot = data['annot']

                            if params.num_gpus == 1:
                                imgs = imgs.cuda()
                                annot = annot.cuda()

                            cls_loss, reg_loss, regression, classification, anchors = model(imgs, annot, obj_list=params.obj_list)
                            cls_loss = cls_loss.mean()
                            reg_loss = reg_loss.mean()

                            loss = cls_loss + reg_loss
                            if loss == 0 or not torch.isfinite(loss):
                                continue

                            loss_classification_ls.append(cls_loss.item())
                            loss_regression_ls.append(reg_loss.item())

                    cls_loss = np.mean(loss_classification_ls)
                    reg_loss = np.mean(loss_regression_ls)
                    loss = cls_loss + reg_loss

                    best_val_model, best_val_loss, current_val_model = \
                        save_model(model, best_val_model, current_val_model, best_val_loss, loss, opt.saved_model_path, step,
                                   opt.compound_coef, opt.mode, val = True)

                    print(
                        'Val. Epoch: {}/{}. Classification loss: {:1.5f}. Regression loss: {:1.5f}. Total loss: {:1.5f}'.format(
                            epoch, opt.num_epochs, cls_loss, reg_loss, loss))

                    loss_writer(opt.saved_path, [cls_loss], [reg_loss], [loss], current_lr, step, epoch, val = True)


                    model.train()

                    # Early stopping
                    if epoch - best_epoch > opt.es_patience > 0:
                        print('[Info] Stop training at epoch {}. The lowest loss achieved is {}'.format(epoch, loss))
                        break


    except KeyboardInterrupt:
        save_checkpoint(model, f'errupted_efficientdet-d{opt.compound_coef}_{epoch}_{step}.pth')


def save_checkpoint(model, name):
    if isinstance(model, CustomDataParallel):
        torch.save(model.module.model.state_dict(), os.path.join(opt.saved_path, name))
    else:
        torch.save(model.model.state_dict(), os.path.join(opt.saved_path, name))




if __name__ == '__main__':
    opt = get_args()
    train(opt)































