#!/bin/bash
#SBATCH --job-name=cuda_matrix_mult
#SBATCH --account=project_2013968
#SBATCH --partition=gpusmall
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=case_%a.txt
#SBATCH --error=case_%a.err
#SBATCH --array=0-9

# Load CUDA module
module load cuda/12.6.1

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

# Create build directory if it doesn't exist
mkdir -p build

# Run CMake to configure the build
cmake -S . -B build

# Compile the code
cmake --build build -j

# Run the executable
srun ./build/app $SLURM_ARRAY_TASK_ID