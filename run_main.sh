#!/bin/bash
#SBATCH --job-name=matmul_app             
#SBATCH --account=project_2013968             
#SBATCH --partition=gpusmall
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --gres=gpu:a100:2
#SBATCH --output=matmul_app_complete.txt

module load cuda

# do all testcases at once
for i in {0..9}; do
    echo "running testcase $i"
    srun ./matmul_app $i
done