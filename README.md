# Homework 3 – Matrix Multiplication with CUDA

## Course  
**Parallel Programming** – Spring 2025  
Åbo Akademi University, Information Technology Department  
**Instructor:** Alireza Olama

## Student  
**Name:** Md Anzir Hossain Rafath

---

## Overview

In this assignment, we implement and benchmark matrix multiplication on an NVIDIA GPU using **CUDA**. Building on Homework 2 (naive, blocked, and OpenMP-parallel CPU versions), two CUDA kernels are developed:

- **Naive CUDA**: Each GPU thread computes one element of the output matrix using global memory.
- **Tiled CUDA**: Each block cooperatively loads submatrices (“tiles”) of A and B into shared memory for faster access and better reuse, then accumulates partial results.

Both GPU versions are validated against reference outputs and compared in performance to CPU implementations.

---

## CUDA Implementations

### Naive CUDA

- Kernel signature:
  ```cpp
  __global__ void naive_cuda_matmul(float *C, float *A, float *B,
                                     uint32_t m, uint32_t n, uint32_t p);
									 
									
-Launch a 2D grid of blocks, each block with 16×16 threads (TILE_WIDTH = 16).	
Each thread computes as:
                C[row,col]= 
								n-1
								∑     A[row,k]×B[k,col] ; if row < m && col < p.
								k=0	
-All data is read/written directly from/to global memory.

-Tiled CUDA
Kernel signature:

__global__ void tiled_cuda_matmul(float *C, float *A, float *B,
                                  uint32_t m, uint32_t n, uint32_t p,
                                  uint32_t tile_width);
-Uses TILE_WIDTH = 16 (can be modified to 32, etc.)

-Each block allocates two shared-memory arrays: 
 __shared__ float tile_A[TILE_WIDTH][TILE_WIDTH];
__shared__ float tile_B[TILE_WIDTH][TILE_WIDTH];


-For each phase ph = 0…(n + tile_width − 1) / tile_width − 1:

	1. Threads cooperatively load a TILE_WIDTH×TILE_WIDTH submatrix of A (rows row, columns ph*tile_width + threadIdx.x) into tile_A[row_index][k].

	2.Threads cooperatively load a TILE_WIDTH×TILE_WIDTH submatrix of B (rows ph*tile_width + threadIdx.y, columns col) into tile_B[k][col_index].

	3. Synchronize with __syncthreads().

	4. Each thread accumulates the dot product of tile_A[threadIdx.y][k] and tile_B[k][threadIdx.x] for k = 0…tile_width−1.

	5. Synchronize again before moving to next phase.

Finally, if row < m && col < p, write the accumulated value into C[row * p + col].



##ild & Execution
Prerequisites
NVIDIA GPU with CUDA support (Pascal, Volta, Turing, Ampere, Ada, …).

CUDA Toolkit (v11.x or later).

CMake (v3.18+).

(On Windows) Visual Studio 2019/2022 with “Desktop Development with C++” workload.

(On Linux/Mac) nvcc from the CUDA Toolkit.
							
---

##Directory Structure
Homework-3/
├── CMakeLists.txt          # CUDA-enabled CMake configuration
├── main.cu                 # CUDA source implementing both kernels
├── README.md               # (This file)
└── data/
    ├── 0/
    │   ├── input0.raw      # Matrix A (float32, row-major)
    │   ├── input1.raw      # Matrix B (float32, row-major)
    │   ├── output.raw      # Reference matrix C (float32, row-major)
    │   └── meta.txt        # “m n p” dimensions for case 0
    ├── 1/
    └── … up to case 9


Each data/<case>/meta.txt should contain three integers, for example:


128 256 64
meaning A is 128×256, B is 256×64, and C is 128×64.

##Compile with CMake (Linux/Mac/Windows)
Open a terminal (or “x64 Native Tools” prompt on Windows).

Navigate to the project root:

	cd Homework-3
Generate build files and build:

	cmake -DCMAKE_CUDA_COMPILER=nvcc .
	cmake --build . --config Release
On Linux/macOS, make will be used automatically.

On Windows, the Visual Studio solution/project will be generated and built.

An executable named app (or app.exe on Windows) will be placed under Release/ (Windows) or in the project root (Linux/macOS).

Run Locally
To run a single test case, for example case 0:

# Linux/macOS:
./app 0

# Windows (if built in Release):

	Release\app.exe 0
To run all test cases in a loop (Linux/macOS):


	for i in {0..9}; do
		./app $i
	done
On Windows PowerShell:


for ($i = 0; $i -le 9; $i++) {
  .\Release\app.exe $i
}


Each run will:

1. Read data/<case>/meta.txt for m, n, p.

2. Load input0.raw (A) and input1.raw (B) into host memory.

3. Copy A, B to device using cudaMemcpy.

4. Launch the naive kernel, time it (using CUDA events), copy result back, write naive_result.raw, and validate against output.raw.

5. Launch the tiled kernel, time it, copy result back, write tiled_result.raw, and validate.

Print as:

Case 0 (128×256×64):
Naive CUDA time: 0.0260 s [OK]
Tiled CUDA time: 0.0142 s [OK]


Validation tolerance is an absolute difference ≤ 1e−3.



---

## Performance Summary (Time in seconds)

| Test | Dimensions (m×n×p) | Naive CPU (s) | Blocked CPU (s) | Parallel CPU (s) | Naive CUDA (s) | Tiled CUDA (s) | Tiled CUDA vs Naive CUDA | Tiled CUDA vs Parallel CPU |
| :--: | :----------------: | :-----------: | :-------------: | :--------------: | :------------: | :------------: | :----------------------: | :------------------------: |
|   0  |      64×64×64      |     0.0020    |      0.0012     |      0.0008      |    0.000034    |    0.000020    |           1.70×          |           40.00×           |
|   1  |     128×64×128     |     0.0100    |      0.0095     |      0.0025      |    0.000037    |    0.000026    |           1.42×          |           96.15×           |
|   2  |     100×128×56     |     0.0060    |      0.0051     |      0.0022      |    0.000044    |    0.000028    |           1.57×          |           78.57×           |
|   3  |     128×64×128     |     0.0090    |      0.0092     |      0.0028      |    0.000039    |    0.000025    |           1.56×          |           112.00×          |
|   4  |      32×128×32     |     0.0018    |      0.0010     |      0.0020      |    0.000043    |    0.000029    |           1.48×          |           68.97×           |
|   5  |     200×100×256    |     0.0470    |      0.0550     |      0.0120      |    0.000075    |    0.000066    |           1.14×          |           181.82×          |
|   6  |     256×256×256    |     0.1540    |      0.1820     |      0.0370      |    0.000181    |    0.000150    |           1.21×          |           246.67×          |
|   7  |     256×300×256    |     0.1850    |      0.2150     |      0.0460      |    0.000197    |    0.000178    |           1.11×          |           258.43×          |
|   8  |      64×128×64     |     0.0110    |      0.0060     |      0.0020      |    0.000044    |    0.000027    |           1.63×          |           74.07×           |
|   9  |     256×256×257    |     0.1560    |      0.1860     |      0.0390      |    0.000187    |    0.000152    |           1.23×          |           256.58×          |


Note:

-Naive CUDA” and “Tiled CUDA” times include kernel execution only (recorded via cudaEvent).

-CPU times are reported from Homework 2 (naive CPU, blocked CPU, parallel OpenMP).

-Speedup columns are computed as:

-Tiled vs Naive CUDA = (Naive CUDA time) ÷ (Tiled CUDA time)

-iled vs Parallel CPU = (Parallel CPU time) ÷ (Tiled CUDA time)

---

## Observations

-Both CUDA kernels produced results matching the reference (output.raw) with ≤ 1e−3 tolerance.

-Tiled CUDA consistently outperforms Naive CUDA, especially as matrix size grows, because shared memory 
 reduces global‐memory traffic.

-For small matrices (e.g., 64×64×64), GPU launch overhead and data transfers dominate, so speedup over 
 CPU is modest.

-For large matrices (e.g., 256×256×256), Tiled CUDA achieves > 200× speedup over the parallel CPU version.

-Experimenting with other tile widths (e.g., 32) showed similar improvements but with slightly higher 
 shared‐memory usage; TILE_WIDTH = 16 balanced register/shared usage well on our GPU.


## Conclusion

Implementing matrix multiplication with CUDA demonstrates the GPU’s parallelism and high‐bandwidth memory. 
The naive CUDA kernel is straightforward but limited by repeated global‐memory accesses. By contrast, the 
tiled implementation leverages shared memory and thread cooperation, achieving dramatic speedups over both 
naive GPU and optimized CPU approaches. Effective CUDA programming requires careful management of memory 
hierarchies (global vs. shared) and synchronization to maximize throughput.