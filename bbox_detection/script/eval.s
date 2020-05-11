#!/bin/bash
#SBATCH --job-name=od_0_eval
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:0

cd ../
python evaluate.py -c 0 --weights 'saved/dl2020/dl2020_0504-193315_coef0_obj_det/model/best-obj_det_efficientdet-d0_20928_val.pth' -th 0.05 --nms_threshold 0.5
