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

### Results
In order to get more objective results, all of these are an average across 5 runs (naive CPU, blocked CPU, parallel CPU, naive CUDA, tiled CUDA, speedups). 
Results for naive CPU, blocked CPU, and parallel CPU I copied from the previous assignment (meaning they were not re-run).
I tested with tile sizes: 16, 32, and 64. I found that there were no significant differences regarding speedup, possibly due to matrix dimensions being small. True speed up differences and memory management come to light only when matrices are big enough to account for the overhead.

I provided a shell script **run_experiments.sh** which will run all 9 cases 5 times and save that to **results.txt**. Shell script is in the root folder, and results file will be created in a new directory called *results*.
I provided **results-assignment2.csv** file with results from the previous assignment in csv format. Next to them we have **results-assignment3.csv** for the results of this assignment. Both results are provided in order to do a join and calculate Tiled CUDA Speedup vs. Parallel CPU. Both are inside of *results* directory.

Lastly, I provided a **results.ipynb** notebook to perform the join of two csv files, calculate Tiled CUDA Speedup vs. Parallel CPU, and finally calculate average metrics. This notebook is also located in the *results* directory.

Tiled CUDA being faster than Naive CUDA is not surprising, as it makes use of shared memory on GPUs, which is much faster than global memory and reduced the overhead of fetching data. Tiled CUDA vs Parallel CPU was a bit surprising, as Tiled CUDA performs worse. That just highlights the bottlenecks of data transfer between CPU and GPU. For cases 6, 7, and 9, which have the biggest matrices, Tiled CUDA still performs worse but only slightly, which again suggest that true speedup power can be seen only with large enough matrices.

I ran these tests on Puhti, and commands that I used are given in the file **puhti_commands**. CMake file had to be configured accordingly (it was giving some errors about platform mismatch).

| Test Case | Dimensions (m × n × p) | Naive CPU (s) | Blocked CPU (s) | Parallel CPU (s) | Naive CUDA (s) | Tiled CUDA (s) | Tiled CUDA Speedup (vs. Naive CUDA) (x) | Tiled CUDA Speedup (vs. Parallel CPU) (x) |
|-----------|------------------------|---------------|-----------------|------------------|----------------|----------------|-------------------------------------|---------------------------------------|
| 0         | 64  × 64  × 64         | 0.001820      | 0.003150        | 0.001971         | 0.147885       | 0.052384       | 2.819076                            | 0.037568                              |
| 1         | 128 × 64  × 128        | 0.007134      | 0.012213        | 0.006315         | 0.179840       | 0.061050       | 2.946002                            | 0.103452                              |
| 2         | 100 x 128 x 56         | 0.004972      | 0.007658        | 0.004033         | 0.155040       | 0.053658       | 2.890278                            | 0.075254                              |
| 3         | 128 x 64  x 128        | 0.007388      | 0.011238        | 0.006139         | 0.182496       | 0.061440       | 2.970274                            | 0.099949                              |
| 4         | 32  x 128 x 32         | 0.000872      | 0.001411        | 0.001214         | 0.144858       | 0.048442       | 2.992014                            | 0.025087                              |
| 5         | 200 x 100 x 256        | 0.035313      | 0.056534        | 0.028929         | 0.281901       | 0.102822       | 2.757510                            | 0.283063                              |
| 6         | 256 × 256 × 256        | 0.118653      | 0.178597        | 0.098928         | 0.311059       | 0.123488       | 2.522300                            | 0.810078                              |
| 7         | 256 × 300 × 256        | 0.138863      | 0.207555        | 0.118014         | 0.299462       | 0.132384       | 2.265972                            | 0.893724                              |
| 8         | 64  x 128 x 64         | 0.003496      | 0.005769        | 0.003319         | 0.138234       | 0.050682       | 2.729630                            | 0.065424                              |
| 9         | 256 x 256 x 257        | 0.120380      | 0.179924        | 0.107247         | 0.294950       | 0.114016       | 2.587976                            | 0.941563                              |