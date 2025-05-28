#!/bin/bash
#SBATCH --job-name=hw3
#SBATCH --account=project_2014289
#SBATCH --partition=gpusmall
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:a100:2
#SBATCH --output=hw3.txt

srun hw3 2
