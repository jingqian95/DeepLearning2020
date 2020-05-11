#!/bin/bash
#SBATCH --job-name=run_test
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mem=4GB
#SBATCH --gres=gpu:1

module purge

cd ../test/
python run_test.py --data_dir '../datasets/dl2020/'