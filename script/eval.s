#!/bin/bash
#SBATCH --job-name=dl2020_eval
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem=8GB
#SBATCH --gres=gpu:0

cd ../
python evaluate.py -c 1 --weights 'saved/dl2020/dl2020_0501-220326_coef1/model/best-efficientdet-d1_25506_val.pth' -th 0.05 --nms_threshold 0.5
