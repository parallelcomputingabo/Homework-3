#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdint.h>     // For uint32_t
#include <iostream>     // std::cout, std::cerr
#include <fstream>      // std::ifstream, std::ofstream
#include <vector>       // std::vector
#include <string>       // std::string
#include <cassert>      // assert
#include <cstring>      // std::memcmp
#include <cmath>        // ceil, floor

#define TILE_WIDTH 16

// Naive matrix multiplication kernel
__global__ void naive_cuda_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < p) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * p + col];
        }
        C[row * p + col] = sum;
    }
}

// Tiled matrix multiplication kernel using shared memory
__global__ void tiled_cuda_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p, uint32_t tile_width) {
    __shared__ float tile_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tile_B[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * tile_width + threadIdx.y;
    int col = blockIdx.x * tile_width + threadIdx.x;

    float sum = 0.0f;
    for (int t = 0; t < (n + tile_width - 1) / tile_width; ++t) {
        if (row < m && t * tile_width + threadIdx.x < n)
            tile_A[threadIdx.y][threadIdx.x] = A[row * n + t * tile_width + threadIdx.x];
        else
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;

        if (t * tile_width + threadIdx.y < n && col < p)
            tile_B[threadIdx.y][threadIdx.x] = B[(t * tile_width + threadIdx.y) * p + col];
        else
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < tile_width; ++k)
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];

        __syncthreads();
    }

    if (row < m && col < p)
        C[row * p + col] = sum;
}

// Compare resulsts to validate code
bool compare_results(float *C1, float *C2, int size, float eps = 1e-3f) {
    for (int i = 0; i < size; ++i)
        if (fabs(C1[i] - C2[i]) > eps)
            return false;
    return true;
}

int main(int argc, char *argv[]) {
    if (argc != 7) {
        printf("Usage: ./app m n p input0.raw input1.raw reference.raw\n");
        return 1;
    }

    // Parse command line arguments
    int m = atoi(argv[1]), n = atoi(argv[2]), p = atoi(argv[3]);
    const char *fileA = argv[4];
    const char *fileB = argv[5];
    const char *fileRef = argv[6];

    // Allocate host memory
    size_t size_A = m * n * sizeof(float);
    size_t size_B = n * p * sizeof(float);
    size_t size_C = m * p * sizeof(float);

    float *h_A = (float *)malloc(size_A);
    float *h_B = (float *)malloc(size_B);
    float *h_C_naive = (float *)malloc(size_C);
    float *h_C_tiled = (float *)malloc(size_C);
    float *h_C_ref = (float *)malloc(size_C);

    // Load input matrices and reference output from files
    FILE *fA = fopen(fileA, "rb");
    FILE *fB = fopen(fileB, "rb");
    FILE *fR = fopen(fileRef, "rb");
    fread(h_A, sizeof(float), m * n, fA); fclose(fA);
    fread(h_B, sizeof(float), n * p, fB); fclose(fB);
    fread(h_C_ref, sizeof(float), m * p, fR); fclose(fR);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    // Copy input matrices to device
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((p + TILE_WIDTH - 1) / TILE_WIDTH, (m + TILE_WIDTH - 1) / TILE_WIDTH);

    // Timing for naive kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    naive_cuda_matmul<<<dimGrid, dimBlock>>>(d_C, d_A, d_B, m, n, p);
    cudaMemcpy(h_C_naive, d_C, size_C, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_naive;
    cudaEventElapsedTime(&time_naive, start, stop);

    // Timing for tiled kernel
    cudaMemset(d_C, 0, size_C);
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    tiled_cuda_matmul<<<dimGrid, dimBlock>>>(d_C, d_A, d_B, m, n, p, TILE_WIDTH);
    cudaMemcpy(h_C_tiled, d_C, size_C, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_tiled;
    cudaEventElapsedTime(&time_tiled, start, stop);

    // Output performance and validation
    printf("Naive CUDA Time: %.6f seconds\n", time_naive / 1000.0);
    printf("Tiled CUDA Time: %.6f seconds\n", time_tiled / 1000.0);
    printf("Naive Validation: %s\n", compare_results(h_C_naive, h_C_ref, m * p) ? "Passed" : "Failed");
    printf("Tiled Validation: %s\n", compare_results(h_C_tiled, h_C_ref, m * p) ? "Passed" : "Failed");

    // Cleanup
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C_naive); free(h_C_tiled); free(h_C_ref);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}