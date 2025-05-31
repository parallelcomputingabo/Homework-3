#include <iostream>
#include <fstream>
#include <string>
#include <cuda_runtime.h>
#include <cstdint>

__global__ void naive_cuda_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p) {
    // Calculate thread indices
    uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check bounds
    if (row < m && col < p) {
        float sum = 0.0f;
        
        // Compute dot product for C[row][col]
        for (uint32_t k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * p + col];
        }
        
        C[row * p + col] = sum;
    }
}

__global__ void tiled_cuda_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p, uint32_t tile_width) {
    // Shared memory for tiles
    extern __shared__ float shared_mem[];
    float* tile_A = shared_mem;
    float* tile_B = &shared_mem[tile_width * tile_width];
    
    // Thread indices
    uint32_t tx = threadIdx.x;
    uint32_t ty = threadIdx.y;
    uint32_t row = blockIdx.y * tile_width + ty;
    uint32_t col = blockIdx.x * tile_width + tx;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (uint32_t tile = 0; tile < (n + tile_width - 1) / tile_width; ++tile) {
        // Load tile of A into shared memory
        uint32_t a_col = tile * tile_width + tx;
        if (row < m && a_col < n) {
            tile_A[ty * tile_width + tx] = A[row * n + a_col];
        } else {
            tile_A[ty * tile_width + tx] = 0.0f;
        }
        
        // Load tile of B into shared memory
        uint32_t b_row = tile * tile_width + ty;
        if (b_row < n && col < p) {
            tile_B[ty * tile_width + tx] = B[b_row * p + col];
        } else {
            tile_B[ty * tile_width + tx] = 0.0f;
        }
        
        // Synchronize to ensure all data is loaded
        __syncthreads();
        
        // Compute partial dot product
        for (uint32_t k = 0; k < tile_width; ++k) {
            sum += tile_A[ty * tile_width + k] * tile_B[k * tile_width + tx];
        }
        
        // Synchronize before loading next tile
        __syncthreads();
    }
    
    // Write result
    if (row < m && col < p) {
        C[row * p + col] = sum;
    }
}

bool validate_result(const std::string &result_file, const std::string &reference_file) {
    // TODO: Implement result validation (same as Assignment 2)
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <case_number>" << std::endl;
        return EXIT_FAILURE;
    }

    int case_number = std::atoi(argv[1]);
    if (case_number < 0 || case_number > 9) {
        std::cerr << "Case number must be between 0 and 9" << std::endl;
        return EXIT_FAILURE;
    }

    // TODO: Read input0.raw (matrix A) and input1.raw (matrix B)

    // TODO: Use cudaMalloc and cudaMemcpy for GPU memory

    // Measure naive CUDA performance
    // TODO: Launch naive_cuda_matmul kernel

    // TODO: Write naive CUDA result to file and validate
    // Measure tiled CUDA performance

    // TODO: Launch tiled_cuda_matmul kernel

    // TODO: Write tiled CUDA result to file and validate

    // Print performance results
    std::cout << "Case " << case_number << " (" << m << "x" << n << "x" << p << "):\n";
    std::cout << "Naive CUDA time: " << naive_cuda_time << " seconds\n";
    std::cout << "Tiled CUDA time: " << tiled_cuda_time << " seconds\n";

    // Clean up

    return 0;
}