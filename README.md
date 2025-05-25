# Parallel Programming Assignment 3: Matrix Multiplication with CUDA

**Åbo Akademi University, Information Technology Department**  
**Instructor:** Alireza Olama
**Submitted By:** Rifat Bin Monsur
**CPU:** Intel i7-11850H with vPRO
**GPU:** NVIDIA RTX A2000

## Overview

This project implements matrix multiplication on the GPU using CUDA.  
The goal is to compare the performance of a naive CUDA implementation (using only global memory) and an optimized tiled CUDA implementation (using shared memory), and measure the speedup achieved by the optimization.

Both CUDA kernels are validated against reference outputs for each test case.

---

## Technical Approach

### Naive CUDA Implementation

- Each CUDA thread computes a single output element in the result matrix.
- All data access is via global memory (no tiling).

### Tiled CUDA Implementation

- Each CUDA thread block computes a TILE_WIDTH x TILE_WIDTH tile of the output.
- Shared memory is used to reduce global memory bandwidth.
- Tile size used: `TILE_WIDTH = 16`.
- Synchronization with `__syncthreads()`.

### File I/O

- Input and output matrices are read and written as text files.
- First line of each file contains matrix dimensions; subsequent lines contain matrix data in row-major order.

---

## How to Build and Run

1. **Clone the repository and place your data folders as specified.**
2. **Build:**
   ```sh
   mkdir build
   cd build
   cmake ..
   cmake --build .
   ```
3. **Run for each test case:**

   ```sh
   .\app.exe 0
   .\app.exe 1
   ...
   .\app.exe 9
   ```
   (Or whatever your executable is called, e.g. main.exe)

   in case the directory shows error, use the following command from root directory:
   ```sh
   .\build\Debug\app.exe 0
   .\build\Debug\app.exe 1
   ...
   .\build\Debug\app.exe 9
   ```

## Results Table

> **Note:**  
> Each CUDA kernel was executed twice per test case. The table below shows the timing for both runs and their average, to reduce noise due to system activity.


| Test Case | Dimensions (m × n × p) | Naive CUDA (s) Run 1 | Naive CUDA (s) Run 2 | **Avg Naive CUDA (s)** | Tiled CUDA (s) Run 1 | Tiled CUDA (s) Run 2 | **Avg Tiled CUDA (s)** | Speedup Run 1 | Speedup Run 2 | **Avg Speedup** |
|-----------|------------------------|----------------------|----------------------|------------------------|----------------------|----------------------|------------------------|---------------|---------------|-----------------|
| 0         | 64×64×64               | 0.00101376           | 0.00013312           | 0.00057344             | 1.2288e-05           | 1.3152e-05           | 1.272e-05              | 82.5          | 10.1217       | 46.31           |
| 1         | 128×64×128             | 0.000171008          | 0.000292864          | 0.000231936            | 1.7408e-05           | 1.7408e-05           | 1.7408e-05              | 9.82353       | 16.8235       | 13.32           |
| 2         | 100×128×56             | 0.000154624          | 0.00016384           | 0.000159232            | 1.8432e-05           | 1.824e-05            | 1.8336e-05              | 8.38889       | 8.98246       | 8.69            |
| 3         | 128×64×128             | 0.000256             | 0.000173056          | 0.000214528            | 1.7408e-05           | 1.7408e-05           | 1.7408e-05              | 14.7059       | 9.94118       | 12.32           |
| 4         | 32×128×32              | 0.000155648          | 0.00013824           | 0.000146944            | 1.536e-05            | 1.5264e-05           | 1.5312e-05              | 10.1333       | 9.0566        | 9.59            |
| 5         | 200×100×256            | 0.00046192           | 0.000448576          | 0.000455248            | 0.000179008          | 0.000137664          | 0.000158336             | 2.58044       | 3.25848       | 2.92            |
| 6         | 256×256×256            | 0.000484544          | 0.000477632          | 0.000481088            | 0.000238848          | 0.000256032          | 0.00024744              | 2.02867       | 1.86552       | 1.95            |
| 7         | 256×300×256            | 0.00062816           | 0.000480192          | 0.000554176            | 0.000238976          | 0.000271072          | 0.000255024             | 2.62855       | 1.77146       | 2.20            |
| 8         | 64×128×64              | 0.000172032          | 0.000156544          | 0.000164288            | 1.536e-05            | 1.4336e-05           | 1.4848e-05              | 11.2          | 10.9196       | 11.06           |
| 9         | 256×256×257            | 0.000537472          | 0.000496672          | 0.000517072            | 0.000253728          | 0.000282656          | 0.000268192             | 2.1183        | 1.75716       | 1.94            |


> **Average Value** 
> Summary table with just the average values for each test case

| Test Case | Dimensions (m × n × p) | Avg Naive CUDA (s) | Avg Tiled CUDA (s) | Avg Speedup |
|-----------|------------------------|--------------------|--------------------|-------------|
| 0         | 64×64×64               | 0.00057344         | 1.272e-05          | 46.31x      |
| 1         | 128×64×128             | 0.000231936        | 1.7408e-05         | 13.32x      |
| 2         | 100×128×56             | 0.000159232        | 1.8336e-05         | 8.69x       |
| 3         | 128×64×128             | 0.000214528        | 1.7408e-05         | 12.32x      |
| 4         | 32×128×32              | 0.000146944        | 1.5312e-05         | 9.59x       |
| 5         | 200×100×256            | 0.000455248        | 0.000158336        | 2.92x       |
| 6         | 256×256×256            | 0.000481088        | 0.00024744         | 1.95x       |
| 7         | 256×300×256            | 0.000554176        | 0.000255024        | 2.20x       |
| 8         | 64×128×64              | 0.000164288        | 1.4848e-05         | 11.06x      |
| 9         | 256×256×257            | 0.000517072        | 0.000268192        | 1.94x       |

## Combined Result Table (CPU and GPU)

| Test Case | Dimensions (m × n × p) | Naive CPU (s) | Blocked CPU (s) | Parallel CPU (s) | Naive CUDA (s) | Tiled CUDA (s) | Tiled CUDA Speedup (vs. Naive CUDA) | Tiled CUDA Speedup (vs. Parallel CPU) |
|-----------|------------------------|---------------|-----------------|------------------|----------------|----------------|--------------------------------------|---------------------------------------|
| 0         | 64×64×64               | 0             | 0.003           | 0.00039          | 0.00057344     | 1.272e-05      | 46.31x                               | 30.66x                                |
| 1         | 128×64×128             | 0             | 0               | 0.0032           | 0.000231936    | 1.7408e-05     | 13.32x                               | 183.80x                               |
| 2         | 100×128×56             | 0.003         | 0               | 0                | 0.000159232    | 1.8336e-05     | 8.69x                                | —                                    |
| 3         | 128×64×128             | 0.0031        | 0               | 0.0032           | 0.000214528    | 1.7408e-05     | 12.32x                               | 183.83x                               |
| 4         | 32×128×32              | 0.0032        |                 | 0.0032           | 0.000146944    | 1.5312e-05     | 9.59x                                | 209.08x                               |
| 5         | 200×100×256            | 0.0032        | 0.00339         | 0.0032           | 0.000455248    | 0.000158336    | 2.92x                                | 20.21x                                |
| 6         | 256×256×256            | 0.0192        | 0.0128          | 0.0032           | 0.000481088    | 0.00024744     | 1.95x                                | 12.94x                                |
| 7         | 256×300×256            | 0.022         | 0.0158          | 0.00359          | 0.000554176    | 0.000255024    | 2.20x                                | 14.08x                                |
| 8         | 64×128×64              | 0             | 0               | 0.00299          | 0.000164288    | 1.4848e-05     | 11.06x                               | 201.39x                               |
| 9         | 256×256×257            | 0.017         | 0.0128          | 0.0034           | 0.000517072    | 0.000268192    | 1.94x                                | 12.68x                                |



## Validation
For all test cases, both CUDA implementations were validated against the provided reference output files and found to be CORRECT.

## Discussion & Reflection

- The tiled CUDA implementation consistently outperformed the naive CUDA kernel, especially on smaller matrices, achieving up to ~80x speedup in some cases.
- On larger matrices, the speedup was still noticeable, but lower (2-3x), likely due to memory access patterns and shared memory limitations.
- For the largest cases, CUDA’s absolute execution times are extremely low (well under a millisecond), highlighting the massive parallel processing power of modern GPUs for matrix operations.
- All results demonstrate the value of shared memory and tiling optimizations in GPU programming.
- The biggest challenge was handling file input/output format correctly (parsing text files and managing matrix dimensions), but once resolved, the CUDA kernels performed as expected.