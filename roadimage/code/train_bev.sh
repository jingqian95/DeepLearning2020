#!/bin/bash

#SBATCH --job-name=road_map_train_val
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10GB
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:k80:1

python train_bev.py 20 /scratch/jq689/deeplearning/DeepLearning2020/code/exp_models/lr_epoch20_final.pt