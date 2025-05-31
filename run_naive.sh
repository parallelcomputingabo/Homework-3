#!/bin/bash
#SBATCH --job-name=naive_gpu
#SBATCH --account=project_2014289
#SBATCH --partition=gpusmall
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=naive_gpu.out

module load cuda/11.5.0

cd ~/Homework-3-main

for i in {0..9}; do
    echo "Running case $i"
    srun ./app data/$i/input0.raw data/$i/input1.raw data/$i/result.raw
done
