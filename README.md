# Parallel Programming Assignment 3: Matrix Multiplication with CUDA

**Åbo Akademi University, Information Technology Department**  
**Instructor:** Alireza Olama  
**Submitted By:** Md Aslam Hossain  
**GPU:** NVIDIA A100 (Mahti cluster)

## Overview

This project implements matrix multiplication on the GPU using CUDA, as part of the Parallel Programming course at Åbo Akademi University. The assignment focuses on comparing the performance of two CUDA implementations:

1. **Naive CUDA**: Uses only global memory, with each thread computing one output element
2. **Tiled CUDA**: Uses shared memory with tiling to reduce global memory access, optimizing performance

Both implementations are validated against provided reference outputs for 10 test cases (0–9), and their execution times are measured to compute the speedup of the tiled implementation over the naive one. The code runs on Mahti's NVIDIA A100 GPUs, leveraging CUDA 12.6.1.

## Technical Approach

### Naive CUDA Implementation

- Each CUDA thread computes a single element in the result matrix **C** (dimensions m×p) by performing a dot product of a row from matrix **A** (m×n) and a column from matrix **B** (n×p)
- All data access is via global memory, leading to higher memory latency
- No optimization for memory coalescing or shared memory usage

### Tiled CUDA Implementation

- Each CUDA thread block computes a TILE_WIDTH × TILE_WIDTH tile of the result matrix **C**
- Shared memory is used to store tiles of matrices **A** and **B**, reducing global memory accesses
- Tile size: TILE_WIDTH = 32, optimized for A100 GPUs
- Synchronization with `__syncthreads()` ensures correct data access within thread blocks
- The number of tiles is computed as `(n + TILE_WIDTH - 1) / TILE_WIDTH`, handling non-divisible dimensions

### File I/O

- **Input**: Matrices **A** and **B** are read from text files (`input0.raw`, `input1.raw`) in each case's data folder (e.g., `data/0/`)
- **Output**: Result matrix **C** is written to `result.raw` with 2 decimal places
- **Reference**: Validation compares `result.raw` against `output.raw` using a tolerance of 10⁻³
- **File format**: First line contains dimensions (rows cols); subsequent lines contain matrix data in row-major order

## How to Build and Run

The project is built using CMake and executed on Mahti's GPU cluster. Follow these steps to replicate the results:

### Prerequisites

- Access to Mahti cluster with CUDA 12.6.1 module
- CMake 3.18 or higher
- Data folders (`data/0/` to `data/9/`) containing `input0.raw`, `input1.raw`, and `output.raw`

### Clone the Repository

```bash
git clone https://github.com/codexaslam/Homework-3/tree/md-aslam-hossain
cd Homework-3
```

### Build

```bash
module load cuda/12.6.1
mkdir build
cd build
cmake ..
cmake --build . -j
```

### Run

Submit the SLURM job array to run all test cases (0–9):

```bash
sbatch ../run_cuda.sh
```

Alternatively, run a single test case manually:

```bash
./app 0
```

## SLURM Script (run_cuda.sh)

The script allocates one A100 GPU, 4GB memory per CPU, and runs cases 0–9:

```bash
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

module load cuda/12.6.1
cd $SLURM_SUBMIT_DIR
mkdir -p build
cmake -S . -B build
cmake --build build -j
srun ./build/app $SLURM_ARRAY_TASK_ID
```

## Performance Results

The following table summarizes the performance for each test case, averaged over multiple runs. Each case was executed 2–5 times, and averages reduce noise due to system variability on Mahti.

| Test Case | Dimensions (m×n×p) | Avg Naive CUDA (s) | Avg Tiled CUDA (s) | Avg Speedup |
|-----------|-------------------|--------------------|--------------------|-------------|
| 0 | 64×64×64 | 0.000186266 | 0.000082144 | 2.27 |
| 1 | 128×64×128 | 0.000213634 | 0.000113664 | 1.88 |
| 2 | 100×128×56 | 0.000220642 | 0.000094528 | 2.33 |
| 3 | 128×64×128 | 0.000213978 | 0.000108134 | 1.98 |
| 4 | 32×128×32 | 0.000189746 | 0.000087954 | 2.16 |
| 5 | 200×100×256 | 0.000309157 | 0.000130186 | 2.37 |
| 6 | 256×256×256 | 0.000368998 | 0.000290592 | 1.27 |
| 7 | 256×300×256 | 0.000382064 | 0.000274808 | 1.39 |
| 8 | 64×128×64 | 0.000209614 | 0.000091376 | 2.29 |
| 9 | 256×256×257 | 0.000369651 | 0.000208876 | 1.77 |

### Performance Notes

- **Naive CUDA times** range from ~159 µs (Case 0) to ~391 µs (Case 7), reflecting increased computational complexity for larger matrices
- **Tiled CUDA times** range from ~78 µs (Case 0) to ~291 µs (Case 6), showing significant improvement
- **Speedups** range from ~1.27x (Case 6) to ~2.37x (Case 5), with higher speedups for smaller and medium-sized matrices due to efficient shared memory usage
- Case 9 (256×256×257) achieves a ~1.77x speedup, indicating good scalability for large matrices

## Validation

All test cases (0–9) were validated against the provided reference outputs (`data/X/output.raw`). Both naive and tiled CUDA implementations produced correct results, as confirmed by:

- `case_X.txt` files reporting `Naive result: PASS` and `Tiled result: PASS`
- Comparison of `result.raw` with `output.raw` within a 10⁻³ tolerance

## Discussion & Reflection

### Performance Analysis

The tiled CUDA implementation consistently outperformed the naive implementation, with speedups ranging from 1.27x to ~2.37x. Smaller matrices (e.g., Case 4: 32×128×32) and medium-sized ones (e.g., Case 5: 200×100×256) benefited most from tiling, achieving ~2.2–2.4x speedups due to efficient shared memory usage. Larger matrices (e.g., Case 6: 256×256×256) showed lower speedups (1.27x), possibly due to increased memory access overhead or suboptimal TILE_WIDTH for these dimensions.

### Scalability

Case 9 (256×256×257) achieved a ~1.77x speedup, demonstrating good scalability for large matrices. The absolute execution times (e.g., ~209 µs tiled for Case 9) highlight the GPU's parallel processing power.

### Optimization Insights

Using TILE_WIDTH = 32 was effective for A100 GPUs, balancing shared memory usage and thread occupancy. Further tuning (e.g., TILE_WIDTH = 16 or 64) could improve performance for specific cases like Case 6.


## Output Files

- `case_X.txt`: Contains timing, validation results, and speedup for case X
- `data/X/result.raw`: Output matrix for case X
- `performance.txt`: Tabulated performance data for all cases
- `case_X.err`: Error logs (empty if no errors)


## Conclusion

This assignment successfully demonstrated the power of GPU acceleration for matrix multiplication using CUDA. The tiled implementation showed significant performance improvements over the naive approach, with an average speedup of ~1.9x across all test cases. The project reinforced key concepts in parallel programming, including memory hierarchy optimization, thread synchronization, and the importance of data locality in high-performance computing.