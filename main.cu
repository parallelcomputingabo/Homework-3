#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

__global__ void naive_cuda_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < p) {
        float sum = 0;
        for (int i = 0; i < n; ++i) {
            sum += A[row * n + i] * B[i * p + col];
        }
        C[row * p + col] = sum;
    }
}

__global__ void tiled_cuda_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p, uint32_t tile_width) {
    __shared__ float tile_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tile_B[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * tile_width + threadIdx.y;
    int col = blockIdx.x * tile_width + threadIdx.x;
    float sum = 0.0f;

    for (int i = 0; i < (n + tile_width - 1) / tile_width; ++i) {
        if (row < m && i * tile_width + threadIdx.x < n)
            tile_A[threadIdx.y][threadIdx.x] = A[row * n + i * tile_width + threadIdx.x];
        else
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;

        if (i * tile_width + threadIdx.y < n && col < p)
            tile_B[threadIdx.y][threadIdx.x] = B[(i * tile_width + threadIdx.y) * p + col];
        else
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int j = 0; j < tile_width; ++j)
            sum += tile_A[threadIdx.y][j] * tile_B[j][threadIdx.x];

        __syncthreads();
    }

    if (row < m && col < p)
        C[row * p + col] = sum;
}

bool validate_result(const std::string &result_file, const std::string &reference_file) {
    std::ifstream result(result_file, std::ios::binary);
    std::ifstream reference(reference_file, std::ios::binary);

    float a, b;
    while (result.read(reinterpret_cast<char*>(&a), sizeof(float)) &&
           reference.read(reinterpret_cast<char*>(&b), sizeof(float))) {
        if (fabs(a - b) > 1e-3) return false;
    }
    return result.eof() && reference.eof();
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: ./matmul <case_number>\n";
        return 1;
    }

    int case_number = std::stoi(argv[1]);
    std::string base_path = "data/" + std::to_string(case_number) + "/";

    uint32_t m, n, p;

    std::ifstream input0(base_path + "input0.raw", std::ios::binary);
    std::ifstream input1(base_path + "input1.raw", std::ios::binary);

    // Read dimensions from first row of input files
    input0.read(reinterpret_cast<char*>(&m), sizeof(uint32_t));
    input0.read(reinterpret_cast<char*>(&n), sizeof(uint32_t));
    input1.read(reinterpret_cast<char*>(&p), sizeof(uint32_t));

    float *A = new float[m * n];
    float *B = new float[n * p];
    float *C_naive = new float[m * p];
    float *C_tiled = new float[m * p];

    input0.read(reinterpret_cast<char*>(A), m * n * sizeof(float));
    input1.seekg(sizeof(uint32_t) * 2); // skip first row
    input1.read(reinterpret_cast<char*>(B), n * p * sizeof(float));

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, m * n * sizeof(float));
    cudaMalloc(&d_B, n * p * sizeof(float));
    cudaMalloc(&d_C, m * p * sizeof(float));

    cudaMemcpy(d_A, A, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, n * p * sizeof(float), cudaMemcpyHostToDevice);

    // Naive
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((p + 15) / 16, (m + 15) / 16);
    naive_cuda_matmul<<<blocksPerGrid, threadsPerBlock>>>(d_C, d_A, d_B, m, n, p);
    cudaDeviceSynchronize();

    cudaMemcpy(C_naive, d_C, m * p * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float naive_time = 0;
    cudaEventElapsedTime(&naive_time, start, stop);

    std::ofstream(base_path + "result_naive.raw", std::ios::binary).write(reinterpret_cast<char*>(C_naive), m * p * sizeof(float));

    // Tiled
    cudaEventRecord(start);
    dim3 threads(TILE_WIDTH, TILE_WIDTH);
    dim3 blocks((p + TILE_WIDTH - 1) / TILE_WIDTH, (m + TILE_WIDTH - 1) / TILE_WIDTH);
    tiled_cuda_matmul<<<blocks, threads>>>(d_C, d_A, d_B, m, n, p, TILE_WIDTH);
    cudaDeviceSynchronize();

    cudaMemcpy(C_tiled, d_C, m * p * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float tiled_time = 0;
    cudaEventElapsedTime(&tiled_time, start, stop);

    std::ofstream(base_path + "result_tiled.raw", std::ios::binary).write(reinterpret_cast<char*>(C_tiled), m * p * sizeof(float));

    std::cout << "Case " << case_number << " (" << m << "x" << n << "x" << p << "):\n";
    std::cout << "Naive CUDA time: " << naive_time / 1000.0f << " seconds\n";
    std::cout << "Tiled CUDA time: " << tiled_time / 1000.0f << " seconds\n";

    bool valid_tiled = validate_result(base_path + "result.raw", base_path + "output.raw");
    std::cout << "Naive Valid: " << (valid_naive ? "Yes" : "No") << "\n";
    std::cout << "Tiled Valid: " << (valid_tiled ? "Yes" : "No") << "\n";

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] A;
    delete[] B;
    delete[] C_naive;
    delete[] C_tiled;

    return 0;
}
