**Parallel Programming**  
**Åbo Akademi University, Information Technology Department**  
**Instructor: Alireza Olama**

**Homework Assignment 3: Matrix Multiplication with CUDA**

**Due Date**: **31/05/2025**  
**Points**: 100

---

### Assignment Overview

Welcome to the third homework assignment of the Parallel Programming course!
In Assignment 2, you optimized matrix multiplication using cache-friendly blocked multiplication and OpenMP for CPU
parallelism. In this assignment, you will take matrix multiplication to the GPU using **CUDA**, NVIDIA’s parallel
computing platform. Your task is to implement matrix multiplication on the GPU, optimize it using CUDA-specific
techniques, and compare its performance with your CPU-based implementations from Assignment 2.

You will implement:

1. **Naive CUDA Matrix Multiplication**: A basic GPU implementation using CUDA kernels.
2. **Tiled CUDA Matrix Multiplication**: An optimized version using shared memory to improve memory access patterns.
3. **Performance Comparison**: Measure and compare the performance of both CUDA implementations against your Assignment
   2 implementations (naive, blocked, and parallel).

This assignment introduces CUDA programming, including kernel launches, thread grids, blocks, and memory management,
while reinforcing the importance of data locality and parallelism.

---

### Technical Requirements

#### 1. Naive CUDA Matrix Multiplication

**Why CUDA?**

CUDA allows you to execute parallel computations on NVIDIA GPUs, which have thousands of cores designed for
data-parallel tasks. Matrix multiplication is an ideal workload for GPUs because it involves independent computations
for each element of the output matrix.

In the naive CUDA implementation, each thread computes one element of the output matrix \( C \). The GPU organizes
threads into a grid of thread blocks, where each block contains a group of threads (e.g., 16x16 threads).

**Naive CUDA Matrix Multiplication**

Assume matrices \( A \) \( m x n \), \( B \) \( n x p \), and \( C \) \( m x p \) are stored in
row-major order in GPU global memory:

```c
__global__ void naive_cuda_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p) {
    
}
```

- **Grid and Block Configuration**: Launch a 2D grid of 2D thread blocks (e.g., 16x16 threads per block).
- **Memory**: Matrices are stored in GPU global memory. Use `cudaMalloc` and `cudaMemcpy` to allocate and transfer data
  between host (CPU) and device (GPU).
- **Task**: Implement the `naive_cuda_matmul` kernel and its host code in the provided `main.cu`. Measure the wall clock
  time, including data transfer times (host-to-device and device-to-host).

#### 2. Tiled CUDA Matrix Multiplication

**Why Tiling?**

The naive CUDA implementation accesses global memory frequently, which is slow (hundreds of cycles per access). CUDA
GPUs have **shared memory**, a fast, on-chip memory shared by threads in a block. Tiled matrix multiplication divides
matrices into tiles (submatrices) that fit into shared memory, reducing global memory accesses and improving
performance.

**Tiled CUDA Matrix Multiplication**

Assume a tile size of `TILE_WIDTH` (e.g., 16 or 32):

```c
__global__ void tiled_cuda_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p, uint32_t tile_width) {

}
```

- **Shared Memory**: Each block loads tiles of \( A \) and \( B \) into shared memory, computes partial results, and
  accumulates the sum.
- **Synchronization**: Use `__syncthreads()` to ensure all threads in a block have loaded data before computation.
- **Task**: Implement the `tiled_cuda_matmul` kernel and its host code in `main.cu`. Experiment with different tile
  sizes (e.g., 16, 32) and report the best performance.

#### 3. Performance Measurement

For each test case (0 through 9, using the same `data` folder from Assignment 2):

- Measure the wall clock time for:
    - **Naive CUDA matrix multiplication** (`naive_cuda_matmul`), including data transfer times.
    - **Tiled CUDA matrix multiplication** (`tiled_cuda_matmul`), including data transfer times.
- Compare with Assignment 2 results (naive, blocked, and parallel CPU implementations).
- Use `cudaEventRecord` and `cudaEventElapsedTime` for accurate GPU timing.
- Report the times in a table in your `README.md`, including:
    - Test case number.
    - Matrix dimensions (\( m \times n \times p \)).
    - Wall clock time for naive CUDA, tiled CUDA, and Assignment 2 implementations (in seconds).
    - Speedup of tiled CUDA over naive CUDA and over Assignment 2’s parallel implementation.

**Example Table Format**:

| Test Case | Dimensions (\( m \times n \times p \)) | Naive CPU (s) | Blocked CPU (s) | Parallel CPU (s) | Naive CUDA (s) | Tiled CUDA (s) | Tiled CUDA Speedup (vs. Naive CUDA) | Tiled CUDA Speedup (vs. Parallel CPU) |
|-----------|----------------------------------------|---------------|-----------------|------------------|----------------|----------------|-------------------------------------|---------------------------------------|
|         |                         |      |           |             |          |          |                               |                                 |

---

### Matrix Storage and Memory Management

- Continue using row-major order for matrices.
- Use CUDA memory management (`cudaMalloc`, `cudaMemcpy`, `cudaFree`) for GPU data.
- Reuse the same input/output format as Assignment 2:
    - Input files: `data/<case>/input0.raw` (matrix \( A \)) and `input1.raw` (matrix \( B \)).
    - Output file: `data/<case>/result.raw` (matrix \( C \)).
    - Reference file: `data/<case>/output.raw` for validation.

---

### Build Instructions

- Use the provided `CMakeLists.txt`, which includes CUDA support.
- **Requirements**:
    - NVIDIA GPU with CUDA support.
    - CUDA Toolkit installed (version 11.x or later recommended).
    - CMake with CUDA language support.
- **Linux/Mac**:
    - Run `cmake -DCMAKE_CUDA_COMPILER=nvcc .` to generate a Makefile, then `make`.
- **Windows**:
    - Use Visual Studio with CUDA toolkit or MinGW with `cmake -G "MinGW Makefiles"`.
- Test with the same test cases (0–9) as Assignment 2.

---

### Submission Requirements

#### Fork and Clone the Repository

- Fork the Assignment 3 repository (provided separately).
- Clone your fork:
  ```bash
  git clone https://github.com/parallelcomputingabo/Homework-3.git
  cd Homework-3
  ```

#### Create a New Branch

```bash
git checkout -b student-name
```

#### Implement Your Solution

- Modify the provided `main.cu` to implement `naive_cuda_matmul` and `tiled_cuda_matmul`.
- Update `README.md` with your performance results table.

#### Commit and Push

```bash
git add .
git commit -m "student-name: Implemented CUDA matrix multiplication"
git push origin student-name
```

#### Submit a Pull Request (PR)

- Create a pull request from your branch to the base repository’s `main` branch.
- Include a description of your CUDA optimizations and any challenges faced.

---

### Grading (100 Points Total)

| Subtask                                       | Points  |
|-----------------------------------------------|---------|
| Correct implementation of `naive_cuda_matmul` | 30      |
| Correct implementation of `tiled_cuda_matmul` | 30      |
| Accurate performance measurements             | 20      |
| Performance results table in `README.md`      | 10      |
| Code clarity, commenting, and organization    | 10      |
| **Total**                                     | **100** |

---

### Tips for Success

- **Naive CUDA**:
    - Ensure correct grid and block dimensions (e.g., `dim3 threadsPerBlock(16, 16)`).
    - Check for CUDA errors using `cudaGetLastError` and `cudaDeviceSynchronize`.
- **Tiled CUDA**:
    - Experiment with tile sizes (e.g., 16, 32) to balance shared memory usage and thread divergence.
    - Minimize shared memory bank conflicts by ensuring contiguous thread access.
- **Performance**:
    - Include data transfer times in measurements, as they are significant for GPU workloads.
    - Run multiple iterations per test case to reduce timing variability.
- **Debugging**:
    - Validate CUDA results against `output.raw` to ensure correctness.
    - Use small matrices for initial testing (e.g., 64x64).
    - Check CUDA documentation for memory management and kernel launch syntax.

---



Good luck, and enjoy accelerating matrix multiplication with CUDA!


Kalle Fjäder Results

These are the Homework 2 results:

| Case | Dimensions (m × n × p) | Naive (s) | Blocked (s) | Parallel (s) | Blocked × | Parallel × |
|------|-----------------------|-----------|-------------|--------------|-----------|------------|
| 0 | 64 × 64 × 64   | 0.0020 | 0.0040 | 0.0020 | 0.50 | 1.00 |
| 1 | 128 × 64 × 128 | 0.0080 | 0.0070 | 0.0030 | 1.14 | 2.67 |
| 2 | 100 × 128 × 56 | 0.0080 | 0.0080 | 0.0020 | 1.00 | 4.00 |
| 3 | 128 × 64 × 128 | 0.0040 | 0.0090 | 0.0040 | 0.44 | 1.00 |
| 4 | 32 × 128 × 32  | 0.0014 | 0.0020 | 0.0014 | 0.69 | 1.00 |
| 5 | 200 × 100 × 256| 0.0240 | 0.0250 | 0.0100 | 0.96 | 2.40 |
| 6 | 256 × 256 × 256| 0.0990 | 0.0800 | 0.0200 | 1.24 | 4.95 |
| 7 | 256 × 300 × 256| 0.0720 | 0.0910 | 0.0230 | 0.79 | 3.13 |
| 8 | 64 × 128 × 64  | 0.0080 | 0.0040 | 0.0020 | 2.00 | 4.00 |
| 9 | 256 × 256 × 257| 0.0590 | 0.1020 | 0.0210 | 0.58 | 2.81 |

Here are the homework 3 results:

| Case | Dimensions (m × n × p) | Naive (s) | Tiled (s) | Parallel (s) | Tiled × | Parallel × |
|------|------------------------|-----------|-----------|--------------|---------|-------------|
| 0    | 64 × 64 × 64           | 0.00129   | 0.00008   | –            | 16.15   | –           |
| 1    | 128 × 64 × 128         | 0.00129   | 0.00004   | –            | 34.70   | –           |
| 2    | 100 × 128 × 56         | 0.00118   | 0.00004   | –            | 29.29   | –           |
| 3    | 128 × 64 × 128         | 0.00130   | 0.00008   | –            | 16.18   | –           |
| 5    | 200 × 100 × 256        | 0.00143   | 0.00008   | –            | 17.60   | –           |
| 6    | 256 × 256 × 256        | 0.00122   | 0.00006   | –            | 20.36   | –           |
| 7    | 256 × 300 × 256        | 0.00124   | 0.00006   | –            | 19.61   | –           |
| 8    | 64 × 128 × 64          | 0.00117   | 0.00004   | –            | 26.89   | –           |
| 9    | 256 × 256 × 257        | 0.00119   | 0.00006   | –            | 20.48   | –           |
