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

## Results

| Test Case | Dimensions (m × n × p) | Naive CPU (s) | Blocked CPU (s) | Parallel CPU (s) | Naive CUDA (s) | Tiled CUDA (s) | Tiled CUDA Speedup (vs. Naive CUDA) | Tiled CUDA Speedup (vs. Parallel CPU) |
|-----------|------------------------|---------------|-----------------|------------------|----------------|----------------|-------------------------------------|---------------------------------------|
| 1         | 64 × 64 × 64           | 0.000771433   | 0.000670545     | 0.000350771      | 0.000048128    | 0.00001856     | 2.59×                               | 18.899×                               |
| 2         | 64 × 64 × 64           | 0.0006139     | 0.000563352     | 0.00023961       | 0.000048128    | 0.00001856     | 2.59×                               | 12.91×                                |
| 3         | 128 × 64 × 128         | 0.0022845     | 0.00227195      | 0.000763971      | 0.000050176    | 0.000017824    | 2.815×                              | 42.86×                                |
| 4         | 128 × 64 × 128         | 0.00295904    | 0.00237161      | 0.0010892        | 0.000050176    | 0.000017824    | 2.815×                              | 60.755×                               |
| 5         | 100 × 128 × 56         | 0.00227332    | 0.00243213      | 0.000535479      | 0.000050176    | 0.000019168    | 2.6177×                             | 27.936×                               |
| 6         | 100 × 128 × 56         | 0.00155518    | 0.00154541      | 0.000579957      | 0.000050176    | 0.000019168    | 2.6177×                             | 28.067×                               |
| 7         | 128 × 64 × 128         | 0.00235209    | 0.00226329      | 0.00071353       | 0.000054272    | 0.0000184      | 2.95×                               | 38.7788×                              |
| 8         | 128 × 64 × 128         | 0.00305947    | 0.00229758      | 0.000697379      | 0.000054272    | 0.0000184      | 2.95×                               | 37.9×                                 |
| 9         | 32 × 128 × 32          | 0.000327104   | 0.000290786     | 0.000148961      | 0.000050176    | 0.000018688    | 2.685×                              | 7.97×                                 |
| 10        | 32 × 128 × 32          | 0.00027893    | 0.000284768     | 0.000204402      | 0.000050176    | 0.000018688    | 2.685×                              | 10.9376×                              |
| 11        | 200 × 100 × 256        | 0.011683      | 0.0114896       | 0.00344001       | 0.000051104    | 0.0000224      | 2.28×                               | 151.79×                               |
| 12        | 200 × 100 × 256        | 0.0117548     | 0.0110553       | 0.00319123       | 0.000051104    | 0.0000224      | 2.28×                               | 142.4656×                             |
| 13        | 256 × 256 × 256        | 0.0383598     | 0.0370295       | 0.00940384       | 0.000066912    | 0.000033504    | 1.997×                              | 280.678×                              |
| 14        | 256 × 256 × 256        | 0.0380587     | 0.0346864       | 0.0095684        | 0.000066912    | 0.000033504    | 1.997×                              | 285.59×                               |
| 15        | 256 × 300 × 256        | 0.0430276     | 0.0413115       | 0.0111705        | 0.000068608    | 0.000038176    | 1.79×                               | 292.605×                              |
| 16        | 256 × 300 × 256        | 0.0419544     | 0.0414222       | 0.0112128        | 0.000068608    | 0.000038176    | 1.79×                               | 293.713×                              |
| 17        | 64 × 128 × 64          | 0.00114947    | 0.00112288      | 0.000451304      | 0.000050144    | 0.00001872     | 2.6786×                             | 24.108×                               |
| 18        | 64 × 128 × 64          | 0.00127682    | 0.00113251      | 0.000446731      | 0.000050144    | 0.00001872     | 2.6786×                             | 23.864×                               |
| 19        | 256 × 256 × 257        | 0.0366115     | 0.0334168       | 0.00918976       | 0.000071360    | 0.000034112    | 2.092×                              | 269.399×                              |
| 20        | 256 × 256 × 257        | 0.0407559     | 0.0336556       | 0.00928626       | 0.000071360    | 0.000034112    | 2.092×                              | 272.228×                              |


