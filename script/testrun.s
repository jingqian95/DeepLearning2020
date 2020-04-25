#!/bin/bash
#SBATCH --job-name=dl2020_test
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00
#SBATCH --mem=8GB
#SBATCH --gres=gpu:2

cd ../
python train.py -n 2