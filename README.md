This file contains the results and details of my work.

I have designed and executed my solution to run on Mahti super computer within the same project of the workshop.


## Performance Results (Including Transfers)

1st table is for averaged results without error check overhead (3 runs) comparing GPU implementations (with transfer times) against Assignment 2 CPU results.

| Test Case | Dimensions (m × n × p) | Naive CPU (s) | Blocked CPU (s) | Parallel CPU (s) | Naive CUDA (s) | Tiled CUDA (s) | Speedup (Tiled vs Naive CUDA) | Speedup (Tiled vs Parallel CPU) |
|-----------|------------------------|---------------|-----------------|------------------|----------------|----------------|-------------------------------|---------------------------------|
| 0         | 64 × 64 × 64           | 0.001784      | 0.001480        | 0.000437         | 0.0001235      | 0.0000768      | 1.61                          | 5.69                            |
| 1         | 128 × 64 × 128         | 0.003957      | 0.003650        | 0.000848         | 0.0001453      | 0.0000971      | 1.50                          | 8.73                            |
| 2         | 100 × 128 × 56         | 0.002369      | 0.002266        | 0.000575         | 0.0001369      | 0.0000875      | 1.56                          | 6.57                            |
| 3         | 128 × 64 × 128         | 0.003344      | 0.003310        | 0.000824         | 0.0001451      | 0.0000973      | 1.49                          | 8.47                            |
| 4         | 32 × 128 × 32          | 0.000434      | 0.000423        | 0.000243         | 0.0001242      | 0.0000763      | 1.63                          | 3.19                            |
| 5         | 200 × 100 × 256        | 0.017642      | 0.015881        | 0.003823         | 0.0002086      | 0.0001674      | 1.25                          | 22.84                           |
| 6         | 256 × 256 × 256        | 0.056921      | 0.053781        | 0.011920         | 0.0002672      | 0.0002242      | 1.19                          | 53.18                           |
| 7         | 256 × 300 × 256        | 0.066703      | 0.064336        | 0.013929         | 0.0002805      | 0.0002473      | 1.13                          | 56.33                           |
| 8         | 64 × 128 × 64          | 0.001738      | 0.001612        | 0.000505         | 0.0001313      | 0.0000856      | 1.53                          | 5.90                            |
| 9         | 256 × 256 × 257        | 0.056782      | 0.053043        | 0.010905         | 0.0002697      | 0.0002334      | 1.16                          | 46.73                           |

2nd table is for averaged results with error check overhead implemented (3 runs) comparing GPU implementations (with transfer times) against Assignment 2 CPU results.

| Case | Size            | Naive CPU (s) | Blocked CPU (s) | Parallel CPU (s) | Naive CUDA (s) | Tiled CUDA (s) | Speedup (vs Naive CUDA) | Speedup (vs Parallel CPU) |
|:----:|:----------------|--------------:|----------------:|-----------------:|---------------:|---------------:|------------------------:|--------------------------:|
| 0   | 64×64×64        | 0.001784      | 0.001480       | 0.000437        | 0.000165       | 0.000081       | 2.02×                   | 5.37×                     |
| 1   | 128×64×128      | 0.003957      | 0.003650       | 0.000848        | 0.000198       | 0.000113       | 1.76×                   | 7.52×                     |
| 2   | 100×128×56      | 0.002369      | 0.002266       | 0.000575        | 0.000187       | 0.000091       | 2.05×                   | 6.32×                     |
| 3   | 128×64×128      | 0.003344      | 0.003310       | 0.000824        | 0.000198       | 0.000117       | 1.69×                   | 7.05×                     |
| 4   | 32×128×32       | 0.000434      | 0.000423       | 0.000243        | 0.000162       | 0.000075       | 2.16×                   | 3.24×                     |
| 5   | 200×100×256     | 0.017642      | 0.015881       | 0.003823        | 0.000284       | 0.000211       | 1.35×                   | 18.12×                    |
| 6   | 256×256×256     | 0.056921      | 0.053781       | 0.011920        | 0.000355       | 0.000277       | 1.28×                   | 42.95×                    |
| 7   | 256×300×256     | 0.066703      | 0.064336       | 0.013929        | 0.000371       | 0.000283       | 1.31×                   | 49.22×                    |
| 8   | 64×128×64       | 0.001738      | 0.001612       | 0.000505        | 0.000175       | 0.000090       | 1.95×                   | 5.61×                     |
| 9   | 256×256×257     | 0.056782      | 0.053043       | 0.010905        | 0.000360       | 0.000279       | 1.29×                   | 39.14×                    |


Note that I got those readings before I add the error-checking code to make my readings consistent with assignment2's, but then I added the debugging/error checking lines. 

For the tiled matrix, I have used TILE_WIDTH of 16, I tried 32 but 16 was slightly faster in most of cases (I think because we have relatively small matrices). 


Naive results go in data/<case>/result.raw, and the tiled version is saved as data/<case>/result_tiled.raw. Neither overwrites the original output.raw.

## Dependencies: 
- CUDA Toolkit ≥ 11.5.0
- NVIDIA A100 or compatible GPU
- CSC's host C++ compiler with its default standard

## Build Instructions:
Refer to the "run_naive.sh" file.

```bash
# Navigate to the project folder
cd ~/Homework-3-main

# Using nvcc directly:
module load cuda/11.5.0
nvcc -O3 -o app main.cu
sbatch run_naive.sh 



### SLURM Job Script (`run_naive_gpu.sh`)
```bash
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

Brief:

Requests a single A100 GPU, 4 CPU cores, and 2 GB RAM per core for up to 30 minutes.

Loads the CUDA 11.5 module, navigates to the project directory, then iterates over test cases 0–9.

The SLURM script passes data/$i/result.raw as the output filename only for the naive kernel, then internally it always writes the tiled output to data/$i/result_tiled.raw without needing a second argument.

Output (stdout/stderr) goes to naive_gpu.out.



## Time calculation: 
“Time with transfers” is to record wall-clock from just before copying A & B Host→Device, through the kernel launch, until after you copy C Device→Host. 
For the tiled matrix time calculation, I had to re-copy A & B from Host -> Device to give a fair comparison. 


Parallelism: 
Block dimensions: 16 × 16 threads

Threads per block: 16 × 16 = 256

Total threads:

m and p be the output matrix’s row and column counts.

Grid size = ⌈p/16⌉ blocks in X × ⌈m/16⌉ blocks in Y

Total blocks = ⌈p/16⌉ × ⌈m/16⌉

Total threads = (⌈p/16⌉ × ⌈m/16⌉) × 256

For example, if m = 64 and p = 64:

Grid = (⌈64/16⌉=4) × (⌈64/16⌉=4) = 4 × 4 blocks

Total blocks = 16

Total threads = 16 × 256 = 4096 threads.





