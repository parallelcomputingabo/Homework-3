# Make sure you're in the right directory
cd ~/Homework-3

# Submit jobs one by one with proper delays
for i in {0..9}; do
  sbatch --export=TEST_CASE=$i run_job.slurm
  sleep 10  # Wait longer between submissions
done