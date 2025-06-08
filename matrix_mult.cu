#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cstdint>
#include <cuda_runtime.h>
#include <iomanip>
#include <sstream>

#define TILE_WIDTH 32
#define BLOCK_SIZE 32

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " code=" << err << "(" << cudaGetErrorName(err) << ") " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

// Naive matrix multiplication kernel
__global__ void naive_matmul(float* C, const float* A, const float* B, 
                            uint32_t m, uint32_t n, uint32_t p) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < p) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * p + col];
        }
        C[row * p + col] = sum;
    }
}

// Optimized tiled matrix multiplication
__global__ void tiled_matmul(float* C, const float* A, const float* B,
                            uint32_t m, uint32_t n, uint32_t p) {
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (n + TILE_WIDTH - 1) / TILE_WIDTH; t++) {
        // Load tiles into shared memory
        if (row < m && t * TILE_WIDTH + tx < n)
            As[ty][tx] = A[row * n + t * TILE_WIDTH + tx];
        else
            As[ty][tx] = 0.0f;
        
        if (col < p && t * TILE_WIDTH + ty < n)
            Bs[ty][tx] = B[(t * TILE_WIDTH + ty) * p + col];
        else
            Bs[ty][tx] = 0.0f;
        
        __syncthreads();
        
        // Compute with loop unrolling
        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < m && col < p)
        C[row * p + col] = sum;
}

// Read binary matrix file
std::vector<float> read_matrix(const std::string& filename, uint32_t& rows, uint32_t& cols) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(1);
    }
    
    file.read(reinterpret_cast<char*>(&rows), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&cols), sizeof(uint32_t));
    
    size_t total = static_cast<size_t>(rows) * cols;
    std::vector<float> matrix(total);
    
    file.read(reinterpret_cast<char*>(matrix.data()), total * sizeof(float));
    
    if (!file) {
        std::cerr << "Error reading matrix data" << std::endl;
        exit(1);
    }
    
    return matrix;
}

// Write binary matrix file
void write_matrix(const std::string& filename, const std::vector<float>& matrix,
                 uint32_t rows, uint32_t cols) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error creating file: " << filename << std::endl;
        exit(1);
    }
    
    file.write(reinterpret_cast<const char*>(&rows), sizeof(uint32_t));
    file.write(reinterpret_cast<const char*>(&cols), sizeof(uint32_t));
    file.write(reinterpret_cast<const char*>(matrix.data()), rows * cols * sizeof(float));
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <test_case>" << std::endl;
        return 1;
    }
    
    int test_case = std::atoi(argv[1]);
    std::string data_dir = "data/" + std::to_string(test_case) + "/";
    
    // Load matrices
    uint32_t m, n, n2, p;
    auto A = read_matrix(data_dir + "input0.raw", m, n);
    auto B = read_matrix(data_dir + "input1.raw", n2, p);
    
    if (n != n2) {
        std::cerr << "Dimension mismatch: A cols (" << n 
                  << ") != B rows (" << n2 << ")" << std::endl;
        return 1;
    }
    
    std::vector<float> C(m * p);
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    size_t size_A = m * n * sizeof(float);
    size_t size_B = n * p * sizeof(float);
    size_t size_C = m * p * sizeof(float);
    
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));
    
    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), size_B, cudaMemcpyHostToDevice));
    
    // Kernel configuration
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((p + threads.x - 1) / threads.x, 
              (m + threads.y - 1) / threads.y);
    
    // Run naive implementation
    naive_matmul<<<grid, threads>>>(d_C, d_A, d_B, m, n, p);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(C.data(), d_C, size_C, cudaMemcpyDeviceToHost));
    write_matrix(data_dir + "result_naive.raw", C, m, p);
    
    // Run tiled implementation
    dim3 tiled_threads(TILE_WIDTH, TILE_WIDTH);
    dim3 tiled_grid((p + tiled_threads.x - 1) / tiled_threads.x,
                    (m + tiled_threads.y - 1) / tiled_threads.y);
    
    tiled_matmul<<<tiled_grid, tiled_threads>>>(d_C, d_A, d_B, m, n, p);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy tiled result back
    CUDA_CHECK(cudaMemcpy(C.data(), d_C, size_C, cudaMemcpyDeviceToHost));
    write_matrix(data_dir + "result_tiled.raw", C, m, p);
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    
    return 0;
}