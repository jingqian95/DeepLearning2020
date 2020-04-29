#!/bin/bash
#SBATCH --job-name=dl2020_eval
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem=8GB
#SBATCH --gres=gpu:0

cd ../
python evaluate.py --weights 'saved/dl2020/dl2020_0428-190731_coef0/model/best-efficientdet-d0_11900.pth'