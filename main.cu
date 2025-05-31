#include <iostream>
#include <fstream>
#include <string>
#include <cuda_runtime.h>
#include <cstdint>

__global__ void naive_cuda_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p) {
    // TODO: Implement naive CUDA matrix multiplication
}

__global__ void tiled_cuda_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p, uint32_t tile_width) {
    // TODO: Implement tiled CUDA matrix multiplication
}

bool validate_result(const std::string &result_file, const std::string &reference_file) {
    // TODO: Implement result validation (same as Assignment 2)
}

int main(int argc, char *argv[]) {


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