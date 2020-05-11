# original author: Zylo117
# adapted from https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch
# modified by muyangjin

"""
Evaluations

put images here datasets/your_project_name/annotations/val_set_name/*.jpg
put annotations here datasets/your_project_name/annotations/instances_{val_set_name}.json
put weights here /path/to/your/weights/*.pth
change compound_coef

"""

import json
import os

import argparse
import torch
import yaml
from tqdm import tqdm
# from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval
import pandas as pd

from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, preprocess_dl

ap = argparse.ArgumentParser()
ap.add_argument('-p', '--project', type=str, default='dl2020', help='project file that contains parameters')
ap.add_argument('-c', '--compound_coef', type=int, default=0, help='coefficients of efficientdet')
ap.add_argument('-w', '--weights', type=str, default=None, help='/path/to/weights')
ap.add_argument('-th', '--threshold', type=float, default=0.05, help='threshold for score')
ap.add_argument('--nms_threshold', type=float, default=0.5,
                help='nms threshold, don\'t change it if not for testing purposes')
ap.add_argument('--cuda', type=bool, default= False)
ap.add_argument('--device', type=int, default=0)
ap.add_argument('--float16', type=bool, default=False)
ap.add_argument('--data_path', type=str, default='datasets/', help='the root folder of dataset')
ap.add_argument('--annotation', type=str, default='annotation_newfeat_2.csv', help='annotation csv file name')
ap.add_argument('--mode', type=str, default='bev', help='bev or ori')

args = ap.parse_args()

compound_coef = args.compound_coef
nms_threshold = args.nms_threshold
use_cuda = args.cuda
gpu = args.device
use_float16 = args.float16
project_name = args.project

weights_path = 'saved/best-efficientdet-d0_7400.pth' if args.weights is None else args.weights

print(f'running coco-style evaluation on project {project_name}, weights {weights_path}...')

params = yaml.safe_load(open(f'projects/{project_name}.yml'))
obj_list = params['obj_list']

input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]

NUM_SAMPLE_PER_SCENE = 126


def evaluate_dl(folder_path, val_index, model, csv_name, mode, threshold=0.05):
    results = pd.DataFrame({'scene_id': [],
                            'sample_id': [],
                            'category_id': [],
                            'score': [],
                            'bbox': []})
    columns = ['scene_id', 'sample_id', 'category_id', 'score', 'bbox']

    # use to transform the output of regresser to boxes
    regressBoxes = BBoxTransform()
    # use to clip the boxes to 0, width/height
    clipBoxes = ClipBoxes()

    ori_imgs, framed_imgs, framed_metas = preprocess_dl(folder_path, val_index, mode, max_size=input_sizes[compound_coef])

    for index in range(len(ori_imgs)):
        scene_id = val_index[0] + index // NUM_SAMPLE_PER_SCENE
        sample_id = index % NUM_SAMPLE_PER_SCENE

        x = torch.from_numpy(framed_imgs[index])

        if use_cuda:
            x = x.cuda(gpu)
            if use_float16:
                x = x.half()
            else:
                x = x.float()
        else:
            x = x.float()

        x = x.unsqueeze(0).permute(0, 3, 1, 2)

        # Run through model
        features, regression, classification, anchors = model(x)

        preds = postprocess(x,
                            anchors, regression, classification,
                            regressBoxes, clipBoxes,
                            threshold, nms_threshold)

        if not preds:
            continue

        preds = invert_affine(framed_metas, preds)[0]

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

                if score < threshold:
                    break

                image_result = pd.Series([scene_id, sample_id, label + 1, float(score), box.tolist()], index=columns)

                results = results.append(image_result, ignore_index = True)

    results.to_csv(csv_name)

    return results


def _eval(coco_gt, image_ids, pred_json_path):
    # load results in COCO evaluation tool
    coco_pred = coco_gt.loadRes(pred_json_path)

    # run COCO evaluation
    print('BBox')
    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


if __name__ == '__main__':
    val_index = range(int(params['val_set'].split(',')[0]), int(params['val_set'].split(',')[1]) + 1)
    folder_path = os.path.join(args.data_path, params['project_name'])
    annotation_file = os.path.join(args.data_path, params['project_name'], args.annotation)
    save_path = os.path.join(weights_path.split('/')[0], weights_path.split('/')[1], weights_path.split('/')[2])
    model_name = weights_path.split('/')[-1].replace('.pth', '')

    csv_name = save_path + '/' + 'evaluation_result_{}_{}_{}_{}_{}.csv'.format(model_name, val_index[0], val_index[-1], args.threshold, args.nms_threshold)

    print(csv_name)
    print(weights_path)
    print(save_path)

    if not os.path.exists(csv_name):
        # Initialize model
        model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                                     ratios=eval(params['anchors_ratios']), scales=eval(params['anchors_scales']))
        # Load weight
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
        model.requires_grad_(False)
        model.eval()
        print('!')

        if use_cuda:
            model.cuda(gpu)

            if use_float16:
                model.half()

        # Run main evaluation
        result_df = evaluate_dl(folder_path, val_index, model, csv_name, args.mode, args.threshold)
        print('CSV file created.')

