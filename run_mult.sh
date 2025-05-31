#!/bin/bash
#SBATCH --job-name=naive_tiled_matrix_mult
#SBATCH --account=project_2013968 
#SBATCH --partition=gpusmall
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=output.txt

srun main 0
srun main 1
srun main 2
srun main 3
srun main 4
srun main 5
srun main 6
srun main 7
srun main 8
srun main 9