cd ~/Homework-3
cat > run_job.slurm << 'EOF'
#!/bin/bash
#SBATCH --job-name=cuda_matmul
#SBATCH --account=project_2013968
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:15:00
#SBATCH --mem=16G
#SBATCH --output=build/job_%j.out
#SBATCH --error=build/job_%j.err

# Load modules properly
source /etc/profile.d/modules.sh
module purge
module use /appl/local/csc/modulefiles
module load gcc/10.4.0 cuda/12.6.1

# Run from build directory
cd build
srun ./main $TEST_CASE
EOF