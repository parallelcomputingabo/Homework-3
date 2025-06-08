#!/bin/bash
#SBATCH --job-name=cuda_matmul
#SBATCH --account=project_2013968 # Replace with your project ID
#SBATCH --partition=gpus
#SBATCH --gres=gpu:v100:1
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=matmul_output_%j.log

module purge
module load gcc/9.4.0 cuda/11.5.0 cmake

cd $SLURM_SUBMIT_DIR

srun ./matrix_mult data/0/input0.raw data/0/input1.raw data/0/result.raw
