#!/bin/bash
#SBATCH --job-name=matmul_cuda
#SBATCH --account=project_2013968
#SBATCH --partition=gpusmall
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=matmul_%j.out
#SBATCH --error=matmul_%j.err




# Run the application with a case number (e.g., 0)
for case in {0..9}; do
    echo "Running case $case"
    ./app $case
done