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

#### Performance Measurement


| Test Case | Dimensions (\( m \times n \times p \)) | Naive CPU (s) | Blocked CPU (s) | Parallel CPU (s) | Naive CUDA (s) | Tiled CUDA (s) | Tiled CUDA Speedup (vs. Naive CUDA) | Tiled CUDA Speedup (vs. Parallel CPU) |
|-----------|----------------------------------------|---------------|-----------------|------------------|----------------|----------------|-------------------------------------|---------------------------------------|
| 0 | 64×64×64 | 0.000771646 | 0.00103871 | 0.000543011 | 0.217056 | 0.00021424 | 1013.15x | 2.54x |
| 1 | 128×64×128 | 0.00322026 | 0.00435753 | 0.000974768 | 0.180308 | 0.00143795 | 125.39x | 0.68x |
| 2 | 100×128×56 | 0.00225026 | 0.00305437 | 0.000868726 | 0.192158 | 0.000141216 | 1360.74x | 6.15x |
| 3 | 128×64×128 | 0.00315414 | 0.00437528 | 0.00110228 | 0.192844 | 0.00144275 | 133.66x | 0.76x |
| 4 | 32×128×32 | 0.000392086 | 0.000540766 | 0.000374269 | 0.202216 | 0.00152822 | 132.32x | 0.24x |
| 5 | 200×100×256 | 0.0227305 | 0.0217888 | 0.0027006 | 0.186828 | 0.000112512 | 1660.52x | 24.00x |
| 6 | 256×256×256 | 0.0561899 | 0.0913475 | 0.0082478 | 0.20308 | 0.00013936 | 1457.23x | 59.19x |
| 7 | 256×300×256 | 0.0676238 | 0.086361 | 0.00924776 | 0.207222 | 0.00150822 | 137.39x | 6.13x |
| 8 | 64×128×64 | 0.0015731 | 0.00206529 | 0.000696908 | 0.201128 | 0.000156928 | 1281.66x | 4.44x |
| 9 | 256×256×257 | 0.0643574 | 0.0748115 | 0.00677648 | 0.185035 | 0.00155738 | 118.81x | 4.35x |

---

### FYI
In previous assignments the way I do the result check is always allowing the floating point deference to be as large as 0.01 (1e-2) to pass the testcases, since I've got results like:
```
Validating results...
Matrix element mismatch at position 21: 977.091 vs 977.09
Matrix element mismatch at position 21: 977.091 vs 977.09
Naive CUDA result is invalid
Tiled CUDA result is invalid
```
So this time I change the validation to:
```c
// Use relative tolerance for better floating-point comparison
float abs_diff = std::abs(r1 - r2);
float rel_diff = abs_diff / (std::abs(r2) + 1e-10f); // Add small epsilon to avoid division by zero
```