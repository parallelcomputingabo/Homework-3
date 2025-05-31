#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <iomanip>  // For formatted output

#define TILE_WIDTH 16

// Read matrix from plain-text .raw file
bool readMatrix(const char* filename, float*& matrix, uint32_t& rows, uint32_t& cols) {
    std::ifstream file(filename);  // TEXT mode (no std::ios::binary)
    if (!file) return false;

    file >> rows >> cols;
    matrix = new float[rows * cols];

    for (uint32_t i = 0; i < rows * cols; ++i) {
        file >> matrix[i];
    }

    file.close();
    return true;
}

// Write matrix to plain-text .raw file
bool writeMatrix(const char* filename, const float* matrix, uint32_t rows, uint32_t cols) {
    std::ofstream file(filename);  // TEXT mode
    if (!file) return false;

    file << rows << " " << cols << "\n";
    file << std::fixed << std::setprecision(6);
    for (uint32_t i = 0; i < rows * cols; ++i) {
        file << matrix[i] << " ";
        if ((i + 1) % cols == 0)
            file << "\n";
    }

    file.close();
    return true;
}

// CUDA error checking macro
#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << "\n"; \
        exit(EXIT_FAILURE); \
    }

// Naive kernel
__global__ void naive_cuda_matmul(float* C, const float* A, const float* B, uint32_t m, uint32_t n, uint32_t p) {
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

// Tiled kernel
__global__ void tiled_cuda_matmul(float* C, const float* A, const float* B, uint32_t m, uint32_t n, uint32_t p, uint32_t tile_width) {
    __shared__ float tile_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tile_B[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * tile_width + threadIdx.y;
    int col = blockIdx.x * tile_width + threadIdx.x;

    float value = 0.0f;
    for (int t = 0; t < (n + tile_width - 1) / tile_width; ++t) {
        if (row < m && t * tile_width + threadIdx.x < n)
            tile_A[threadIdx.y][threadIdx.x] = A[row * n + t * tile_width + threadIdx.x];
        else
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < p && t * tile_width + threadIdx.y < n)
            tile_B[threadIdx.y][threadIdx.x] = B[(t * tile_width + threadIdx.y) * p + col];
        else
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < tile_width; ++k)
            value += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];

        __syncthreads();
    }

    if (row < m && col < p)
        C[row * p + col] = value;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: ./app input0.raw input1.raw output.raw\n";
        return 1;
    }

    float *h_A = nullptr, *h_B = nullptr, *h_C = nullptr;
    uint32_t m, n, n2, p;

    if (!readMatrix(argv[1], h_A, m, n)) {
        std::cerr << "Failed to read matrix A\n";
        return 1;
    }
    if (!readMatrix(argv[2], h_B, n2, p)) {
        std::cerr << "Failed to read matrix B\n";
        delete[] h_A;
        return 1;
    }
    if (n != n2) {
        std::cerr << "Matrix dimension mismatch\n";
        delete[] h_A;
        delete[] h_B;
        return 1;
    }

    h_C = new float[m * p];

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, m * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, n * p * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, m * p * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, m * n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, n * p * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // --- Naive kernel ---
    CUDA_CHECK(cudaEventRecord(start));
    dim3 threadsPerBlock_naive(16, 16);
    dim3 blocksPerGrid_naive((p + 15) / 16, (m + 15) / 16);
    naive_cuda_matmul<<<blocksPerGrid_naive, threadsPerBlock_naive>>>(d_C, d_A, d_B, m, n, p);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float timeNaive = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&timeNaive, start, stop));

    CUDA_CHECK(cudaMemcpy(h_C, d_C, m * p * sizeof(float), cudaMemcpyDeviceToHost));
    writeMatrix(argv[3], h_C, m, p);

    std::cout << "Naive CUDA kernel time (ms): " << timeNaive << "\n";

    // --- Tiled kernel ---
    CUDA_CHECK(cudaMemset(d_C, 0, m * p * sizeof(float)));
    CUDA_CHECK(cudaEventRecord(start));

    dim3 threadsPerBlock_tiled(TILE_WIDTH, TILE_WIDTH);
    dim3 blocksPerGrid_tiled((p + TILE_WIDTH - 1) / TILE_WIDTH, (m + TILE_WIDTH - 1) / TILE_WIDTH);
    tiled_cuda_matmul<<<blocksPerGrid_tiled, threadsPerBlock_tiled>>>(d_C, d_A, d_B, m, n, p, TILE_WIDTH);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float timeTiled = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&timeTiled, start, stop));

    CUDA_CHECK(cudaMemcpy(h_C, d_C, m * p * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "Tiled CUDA kernel time (ms): " << timeTiled << "\n";

    // Cleanup
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
