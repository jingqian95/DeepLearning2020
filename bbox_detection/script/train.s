#!/bin/bash
#SBATCH --job-name=od_1_train
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=16:00:00
#SBATCH --mem=8GB
#SBATCH --gres=gpu:2

module purge

cd ../
python train.py -c 2