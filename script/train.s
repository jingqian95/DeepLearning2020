#!/bin/bash
#SBATCH --job-name=dl2020_test
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:2

module purge

cd ../
python train.py -c 3