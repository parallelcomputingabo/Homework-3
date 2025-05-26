#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <cassert>
#include <filesystem>
#include <iomanip>
#include <chrono>
#include <cmath>

#define TILE_WIDTH 16

__global__ void naive_cuda_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p) {
    // TODO: Implement naive CUDA matrix multiplication
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < p) {
        float value = 0.0f;
        for (int i = 0; i < n; ++i) {
            value += A[row * n + i] * B[i * p + col];
        }
        C[row * p + col] = value;
    }
}

__global__ void tiled_cuda_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p, uint32_t tile_width) {
    // TODO: Implement tiled CUDA matrix multiplication
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
        for (int i = 0; i < tile_width; ++i)
            value += tile_A[threadIdx.y][i] * tile_B[i][threadIdx.x];
        __syncthreads();
    }
    if (row < m && col < p){
        C[row * p + col] = value;
    }
}

bool validate_result(const std::string &result_file, const std::string &reference_file) {
    // TODO: Implement result validation (same as Assignment 2)
    std::ifstream res(result_file);
    std::ifstream ref(reference_file);
    if (!res.is_open() || !ref.is_open()) {
        return false;
    }
    uint32_t rm, rp, cm, cp;
    res >> rm >> rp;
    ref >> cm >> cp;
    if (rm != cm || rp != cp) return false;
    float vres, vref;
    const float eps = 1e-6f;
    for (uint32_t i = 0; i < rm * rp; ++i) {
        if (!(res >> vres) || !(ref >> vref)) return false;
        if (std::fabs(vres - vref) > eps) {
            return false;
        }
    }
    return true;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <test_case_number>\n";
        return 1;
    }

    int case_number = std::stoi(argv[1]);
    std::string path_prefix = "data/" + std::to_string(case_number) + "/";
    std::string input0_file = path_prefix + "input0.raw";
    std::string input1_file = path_prefix + "input1.raw";
    std::string output_file = path_prefix + "result.raw";
    std::string reference_file = path_prefix + "output.raw";

    uint32_t m, n, n2, p;
    std::vector<float> h_A, h_B;

    // Open and read matrix A from file
    std::ifstream inA(input0_file);
    inA >> m >> n;
    h_A.resize(m * n);
    for (uint32_t i = 0; i < m * n; ++i) inA >> h_A[i];
    inA.close();

    // Open and read matrix B from file
    std::ifstream inB(input1_file);
    inB >> n2 >> p;
    if (n2 != n) return 1;
    h_B.resize(n * p);
    for (uint32_t i = 0; i < n * p; ++i) inB >> h_B[i];
    inB.close();

    std::vector<float> h_C_naive(m * p);
    std::vector<float> h_C_tiled(m * p);
    float *d_A, *d_B, *d_C;

    // Allocate device memory
    cudaMalloc(&d_A, m * n * sizeof(float));
    cudaMalloc(&d_B, n * p * sizeof(float));
    cudaMalloc(&d_C, m * p * sizeof(float));

    // Copy matrices A and B to device memory
    cudaMemcpy(d_A, h_A.data(), m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), n * p * sizeof(float), cudaMemcpyHostToDevice);

    // Define CUDA kernel dimensions
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((p + TILE_WIDTH - 1) / TILE_WIDTH, (m + TILE_WIDTH - 1) / TILE_WIDTH);

    // Measure performance of naive CUDA kernel
    cudaEvent_t start_naive, stop_naive, start_tiled, stop_tiled;
    float naive_time = 0, tiled_time = 0;
    cudaEventCreate(&start_naive); cudaEventCreate(&stop_naive);
    cudaEventCreate(&start_tiled); cudaEventCreate(&stop_tiled);

    cudaEventRecord(start_naive);
    naive_cuda_matmul<<<gridDim, blockDim>>>(d_C, d_A, d_B, m, n, p);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C_naive.data(), d_C, m * p * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop_naive);
    cudaEventSynchronize(stop_naive);
    cudaEventElapsedTime(&naive_time, start_naive, stop_naive);

    // Measure performance of tiled CUDA kernel
    cudaEventRecord(start_tiled);
    tiled_cuda_matmul<<<gridDim, blockDim>>>(d_C, d_A, d_B, m, n, p, TILE_WIDTH);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C_tiled.data(), d_C, m * p * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop_tiled);
    cudaEventSynchronize(stop_tiled);
    cudaEventElapsedTime(&tiled_time, start_tiled, stop_tiled);

    std::ofstream result_file(output_file);
    result_file << m << " " << p << "\n";
    for (uint32_t i = 0; i < m; ++i) {
        for (uint32_t j = 0; j < p; ++j)
            result_file << h_C_tiled[i * p + j] << " ";
        result_file << "\n";
    }
    result_file.close();


    // Validate correctness against reference output
    bool valid_naive = validate_result("data/" + std::to_string(case_number) + "/result_naive.raw", reference_file);
    bool valid_tiled = validate_result(output_file, reference_file);

    // Output final performance results
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Test Case: " << case_number << "\n";
    std::cout << "Dimensions (( m \\times n \\times p )): " << m << " x " << n << " x " << p << "\n";
    std::cout << "Naive CUDA: " << naive_time / 1000.0f << " s\n";
    std::cout << "Tiled CUDA: " << tiled_time / 1000.0f << " s\n";
    std::cout << "Tiled CUDA Speedup (vs Naive CUDA): " << (naive_time / tiled_time) << "x\n";

    // Cleanup device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}