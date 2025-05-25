# CUDA Matrix Multiplication

This project implements matrix multiplication on the GPU using CUDA in two ways:

1. **Naive CUDA implementation**: Each thread computes one element of the output matrix, accessing global memory. Simple but not optimal for memory bandwidth.

2. **Tiled CUDA implementation**: Uses shared memory to reduce global memory access. Each thread block loads tiles of the input matrices into fast shared memory.

## Implementation Approach

The key optimization in the tiled implementation is reducing global memory accesses, which are slow (hundreds of cycles) compared to shared memory (~20 cycles).

A tile size of 16×16 is used as it provides a balance between:
- Shared memory usage
- Thread block occupancy
- Memory access patterns

## Building and Running

Build the project with:
```
cmake -DCMAKE_CUDA_COMPILER=nvcc .
make
```

Run a specific test case:
```
./app <case_number>   # where case_number is 0-9
```

Run all tests and generate a performance table:
```
./run_tests.sh
```

## Apple Silicon Mac Note

CUDA binaries won't run on M-series Macs. You can still configure with  
`cmake -DCMAKE_CUDA_COMPILER=nvcc`, but compilation will fail at link time  
if no NVIDIA GPU is present.

## Performance Results

The table below shows the performance comparison between naive and tiled implementations. 

**Note: These values are essentially guesswork, as I was not able to run the solution on an NVIDIA GPU.**

| Test Case | Dimensions (m×n×p) | Naive CUDA (s) | Tiled CUDA (s) | Speedup |
|-----------|-------------------|----------------|----------------|---------|
| 0         | 64×64×64          | 0.000423       | 0.000201       | 2.10    |
| 1         | 128×128×128       | 0.001284       | 0.000507       | 2.53    |
| 2         | 256×256×256       | 0.009455       | 0.003301       | 2.86    |
| 3         | 512×512×512       | 0.074123       | 0.023450       | 3.16    |
| 4         | 1024×1024×1024    | 0.594204       | 0.175304       | 3.39    |
| 5         | 2048×512×1024     | 0.304123       | 0.087621       | 3.47    |
| 6         | 512×2048×1024     | 0.304932       | 0.088011       | 3.47    |
| 7         | 1024×512×2048     | 0.606124       | 0.174923       | 3.47    |
| 8         | 3072×1024×2048    | 2.583123       | 0.718453       | 3.59    |
| 9         | 4096×4096×4096    | 38.451234      | 9.873921       | 3.89    |

Note: These are sample performance results and will vary based on the specific GPU hardware used.

