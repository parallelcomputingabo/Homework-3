# Homework 3 – Matrix Multiplication with CUDA

## Course  
**Parallel Programming – Spring 2025**  
Åbo Akademi University

## Student  
**Name:** Fahmida Khalid

---

## Overview

This project implements matrix multiplication using **NVIDIA CUDA** on the GPU. Two CUDA versions are developed:

- **Naive CUDA:** Each thread computes one element of the output matrix using only global memory.
- **Tiled CUDA:** Each thread block loads tiles of input matrices into **shared memory** for faster memory access and better performance.

These GPU implementations are compared against CPU versions (Naive, Blocked, and OpenMP) from Homework 2.

---

## CUDA Implementations

### Naive CUDA

- 2D grid of 16×16 threads per block.
- Each thread computes one output element directly using global memory.
- Simple but inefficient due to slow memory access.

### Tiled CUDA

- Uses TILE_WIDTH = 16.
- Loads submatrices (tiles) of A and B into shared memory.
- Threads in a block cooperate and use `__syncthreads()` to synchronize.
- Reduces global memory usage and improves performance.

---

## Matrix Sizes Tested

| Test Case | Dimensions (m × n × p) |
|-----------|------------------------|
| 0         | 64 × 64 × 64           |
| 1         | 128 × 64 × 128         |
| 2         | 100 × 128 × 56         |
| 3         | 128 × 64 × 128         |
| 4         | 32 × 128 × 32          |
| 5         | 200 × 100 × 256        |
| 6         | 256 × 256 × 256        |
| 7         | 256 × 300 × 256        |
| 8         | 64 × 128 × 64          |
| 9         | 256 × 256 × 257        |

---

## Compilation and Execution (Google Colab )

```bash
# Setup Environment
!apt update
!apt install -y cmake ninja-build nvidia-cuda-toolkit
!nvcc --version
!nvidia-smi

# Build
!cmake -S Homework-3 -B Homework-3/build -G "Unix Makefiles"
!cmake --build Homework-3/build

# Run All Test Cases
!for i in {0..9}; do ./Homework-3/build/app Homework-3/data/$i/input0.raw Homework-3/data/$i/input1.raw Homework-3/data/$i/result.raw; done
```

---

## Performance Summary (Time in milliseconds)

| Test | Dimensions       | Naive CPU | Blocked CPU | Parallel CPU | Naive CUDA | Tiled CUDA | Tiled vs Naive | Tiled vs Parallel |
|------|------------------|-----------|-------------|--------------|------------|------------|----------------|-------------------|
| 0    | 64×64×64         | 2         | 3           | 1            | 0.0344     | 0.0203     | 1.69×          | 49.26×            |
| 1    | 128×64×128       | 10        | 11          | 3            | 0.0369     | 0.0266     | 1.39×          | 112.78×           |
| 2    | 100×128×56       | 6         | 8           | 3            | 0.0445     | 0.0276     | 1.61×          | 108.70×           |
| 3    | 128×64×128       | 9         | 11          | 3            | 0.0389     | 0.0255     | 1.53×          | 117.65×           |
| 4    | 32×128×32        | 2         | 1           | 2            | 0.0430     | 0.0287     | 1.50×          | 69.69×            |
| 5    | 200×100×256      | 47        | 55          | 12           | 0.0750     | 0.0661     | 1.13×          | 181.55×           |
| 6    | 256×256×256      | 154       | 182         | 37           | 0.1815     | 0.1503     | 1.21×          | 246.12×           |
| 7    | 256×300×256      | 185       | 215         | 46           | 0.1970     | 0.1782     | 1.11×          | 258.21×           |
| 8    | 64×128×64        | 11        | 6           | 2            | 0.0442     | 0.0273     | 1.62×          | 73.26×            |
| 9    | 256×256×257      | 156       | 186         | 39           | 0.1869     | 0.1522     | 1.23×          | 256.22×           |

---

## Observations

- CUDA outputs were verified and matched expected results.
-  Tiled CUDA is consistently faster than naive CUDA due to efficient memory usage.
- CUDA implementations outperform CPU (even parallel OpenMP) especially for large matrices.
-  Small matrices show less speedup due to overhead from kernel launch and data transfer.

---

## Conclusion

CUDA provides dramatic performance improvements for matrix multiplication, especially when shared memory is used efficiently. The tiled CUDA kernel offers substantial speedup over both the naive GPU kernel and optimized CPU implementations. Effective GPU programming requires optimizing memory access and leveraging the hierarchical memory architecture of CUDA-capable devices.
