#!/bin/bash
#SBATCH --job-name=rd_train
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:2

module purge

cd ../
python train.py --mode 'roadimage'