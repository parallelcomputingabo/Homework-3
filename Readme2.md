# CUDA Matrix Multiplication - Homework Assignment 3

## Student Information
- **Course**: Parallel Programming, Åbo Akademi University
- **Assignment**: Homework 3 - Matrix Multiplication with CUDA
- **Due Date**: 31/05/2025

## Implementation Overview

This assignment implements two CUDA matrix multiplication algorithms:

1. **Naive CUDA Matrix Multiplication**: Each thread computes one element of the output matrix using global memory access
2. **Tiled CUDA Matrix Multiplication**: Uses shared memory to optimize memory access patterns by loading tiles of matrices into fast shared memory

## Key Features

- **Error Handling**: Comprehensive CUDA error checking with informative error messages
- **Memory Management**: Proper GPU memory allocation, transfer, and cleanup
- **Performance Measurement**: Accurate timing using CUDA events
- **Result Validation**: Automatic validation against reference output files
- **Optimized Compilation**: Fast math optimizations and appropriate CUDA architectures

## Technical Implementation Details

### Naive CUDA Implementation
- **Thread Organization**: 2D grid of 16x16 thread blocks
- **Memory Access**: Direct global memory access for matrices A and B
- **Computation**: Each thread computes C[row][col] = Σ(A[row][k] * B[k][col])

### Tiled CUDA Implementation
- **Tile Size**: 16x16 (configurable via TILE_WIDTH macro)
- **Shared Memory**: Each block loads tiles into fast shared memory
- **Synchronization**: Uses __syncthreads() for proper thread coordination
- **Memory Optimization**: Reduces global memory accesses by factor of tile_width

## Performance Results

| Test Case | Dimensions (m×n×p) | Naive CPU (s) | Blocked CPU (s) | Parallel CPU (s) | Naive CUDA (s) | Tiled CUDA (s) | Tiled CUDA Speedup (vs. Naive CUDA) | Tiled CUDA Speedup (vs. Parallel CPU) |
|-----------|-------------------|---------------|-----------------|------------------|----------------|----------------|--------------------------------------|---------------------------------------|
| 0         | TBD              | N/A           | N/A             | N/A              | TBD            | TBD            | TBD                                  | N/A                                   |
| 1         | TBD              | N/A           | N/A             | N/A              | TBD            | TBD            | TBD                                  | N/A                                   |
| 2         | TBD              | N/A           | N/A             | N/A              | TBD            | TBD            | TBD                                  | N/A                                   |
| 3         | TBD              | N/A           | N/A             | N/A              | TBD            | TBD            | TBD                                  | N/A                                   |
| 4         | TBD              | N/A           | N/A             | N/A              | TBD            | TBD            | TBD                                  | N/A                                   |
| 5         | TBD              | N/A           | N/A             | N/A              | TBD            | TBD            | TBD                                  | N/A                                   |
| 6         | TBD              | N/A           | N/A             | N/A              | TBD            | TBD            | TBD                                  | N/A                                   |
| 7         | TBD              | N/A           | N/A             | N/A              | TBD            | TBD            | TBD                                  | N/A                                   |
| 8         | TBD              | N/A           | N/A             | N/A              | TBD            | TBD            | TBD                                  | N/A                                   |
| 9         | TBD              | N/A           | N/A             | N/A              | TBD            | TBD            | TBD                                  | N/A                                   |

*Note: CPU timing results from Assignment 2 are not available. This table will be updated with actual performance measurements.*

## Build Instructions

### Prerequisites
- NVIDIA GPU with CUDA support
- CUDA Toolkit (version 11.x or later)
- CMake (version 3.18 or later)
- GCC compiler

### Building on CSC Mahti

1. **Load required modules**:
```bash
module purge
module load gcc/11.3.0
module load cuda/11.7.0
module load cmake/3.24.2
```

2. **Build the project**:
```bash
cmake -DCMAKE_CUDA_COMPILER=nvcc .
make
```

3. **Run tests**:
```bash
# Run single test case
./matrix_mult <test_case_number>

# Run all test cases
chmod +x run_all_tests.sh
./run_all_tests.sh
```

### Running on CSC Mahti with SLURM

1. **Submit batch job**:
```bash
# Edit submit_job.sh to use your project account
sbatch submit_job.sh
```

2. **Monitor job**:
```bash
squeue -u $USER
```

3. **Check results**:
```bash
cat cuda_matmul_<jobid>.out
```

## File Structure

```
.
├── main.cu                 # Main CUDA implementation
├── CMakeLists.txt         # Build configuration
├── submit_job.sh          # SLURM batch script
├── run_all_tests.sh       # Test runner script
├── README.md              # This documentation
└── data/                  # Test data directory
    ├── 0/
    │   ├── input0.raw     # Matrix A
    │   ├── input1.raw     # Matrix B
    │   ├── output.raw     # Reference result
    │   └── result.raw     # Generated result
    └── ...
```

## Key Optimizations Implemented

1. **Memory Coalescing**: Ensured contiguous memory access patterns for optimal GPU memory bandwidth
2. **Shared Memory Usage**: Utilized fast shared memory to reduce global memory accesses
3. **Thread Block Organization**: Optimized thread block dimensions for GPU occupancy
4. **Bank Conflict Avoidance**: Structured shared memory access to minimize bank conflicts
5. **Fast Math Operations**: Enabled CUDA fast math optimizations for improved performance

## Challenges and Solutions

1. **Memory Management**: Implemented comprehensive error checking for all CUDA memory operations
2. **Boundary Conditions**: Properly handled matrix dimensions that don't align with tile sizes
3. **Synchronization**: Correctly used __syncthreads() to ensure data consistency in shared memory
4. **Performance Measurement**: Used CUDA events for accurate GPU timing excluding CPU overhead

## Expected Performance Characteristics

- **Tiled implementation** should significantly outperform naive implementation for larger matrices
- **Memory-bound operations** will show greater improvement with tiling optimization
- **GPU vs CPU** performance will depend on matrix size and memory transfer overhead

## Validation

All implementations are validated against reference output files to ensure correctness. The validation uses a tolerance of 1e-4 for floating-point comparisons.