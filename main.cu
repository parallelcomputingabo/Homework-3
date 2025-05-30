#include <iostream>
#include <fstream>
#include <string>
#include <cuda_runtime.h>
#include <vector>
#include <cassert>
#include <filesystem>
#include <iomanip>


#define TILE_WIDTH 16 // 16 seemed more consistant than 32

__global__ void naive_cuda_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p) {
    // TODO: Implement naive CUDA matrix multiplication
    uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < p) {
        float sum = 0.0f;
        for (uint32_t i = 0; i < n; ++i) {
            sum += A[row * n + i] * B[i * p + col];
        }
        C[row * p + col] = sum;
    }
}

__global__ void tiled_cuda_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p, uint32_t tile_width) {
    // TODO: Implement tiled CUDA matrix multiplication
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    uint32_t row = blockIdx.y * tile_width + threadIdx.y;
    uint32_t col = blockIdx.x * tile_width + threadIdx.x;

    float sum = 0.0f;

    for (uint32_t tile = 0; tile < (n + tile_width - 1) / tile_width; ++tile) {

        uint32_t a_col = tile * tile_width + threadIdx.x;
        uint32_t b_row = tile * tile_width + threadIdx.y;

        if (row < m && a_col < n) {
            As[threadIdx.y][threadIdx.x] = A[row * n + a_col];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (b_row < n && col < p) {
            Bs[threadIdx.y][threadIdx.x] = B[b_row * p + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (uint32_t i = 0; i < tile_width; ++i) {
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < m && col < p) {
        C[row * p + col] = sum;
    }
}

bool validate_result(const std::string &result_file, const std::string &reference_file) {
    // TODO: Implement result validation (same as Assignment 2)
    std::ifstream res(result_file);
    std::ifstream ref(reference_file);

    if (!res.is_open() || !ref.is_open()) {
        return false;
    }

    // Introduced some tolerance since I don't seem to solve an issue with
    // getting the decimals written the same as the output.
    float tolerance = 1e-2f;
    float val1, val2;
    while (res >> val1 && ref >> val2) {
        if (std::abs(val1 - val2) > tolerance) {
            return false;
        }
    }

    // Check if one file had more lines
    if (res.eof() != ref.eof()) {
        return false;
    }

    return true;
}

int main(int argc, char *argv[]) {
    int case_number = std::stoi(argv[1]);


    // TODO: Read input0.raw (matrix A) and input1.raw (matrix B)
    std::string folder = "../data/" + std::to_string(case_number) + "/";
    std::ifstream a_file(folder + "input0.raw");
    std::ifstream b_file(folder + "input1.raw");
    uint32_t m, n, p;
    float *A, *B, *C_naive, *C_tiled;


    a_file >> m >> n;
    b_file >> n >> p;

    A = new float[m * n];
    B = new float[n * p];

    for (uint32_t i = 0; i < m * n; ++i) {
        a_file >> A[i];
    }
    for (uint32_t i = 0; i < n * p; ++i) {
        b_file >> B[i];
    }

    a_file.close();
    b_file.close();


    // TODO: Use cudaMalloc and cudaMemcpy for GPU memory
    C_naive = new float[m * p];
    C_tiled = new float[m * p];
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, m * n * sizeof(float));
    cudaMalloc(&d_B, n * p * sizeof(float));
    cudaMalloc(&d_C, m * p * sizeof(float));

    cudaMemcpy(d_A, A, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, n * p * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((p + TILE_WIDTH - 1) / TILE_WIDTH, (m + TILE_WIDTH - 1) / TILE_WIDTH);

    cudaEvent_t start_naive, stop_naive, start_tiled, stop_tiled;
    cudaEventCreate(&start_naive);
    cudaEventCreate(&stop_naive);
    cudaEventCreate(&start_tiled);
    cudaEventCreate(&stop_tiled);

    // Measure naive CUDA performance
    // TODO: Launch naive_cuda_matmul kernel
    cudaEventRecord(start_naive);
    naive_cuda_matmul<<<dimGrid, dimBlock>>>(d_C, d_A, d_B, m, n, p);
    cudaDeviceSynchronize();

    cudaMemcpy(C_naive, d_C, m * p * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop_naive);
    cudaEventSynchronize(stop_naive);
    float naive_cuda_time = 0;
    cudaEventElapsedTime(&naive_cuda_time, start_naive, stop_naive);
    naive_cuda_time = naive_cuda_time / 1000; // Since the time is recorded in milliseconds

    // TODO: Write naive CUDA result to file and validate

    std::ofstream result_file(folder + "result_naive.raw");

    result_file << m << " " << p << std::endl;
    for (int i = 0; i < m;result_file<<std::endl, i++) {
        for (int j = 0; j < p; j++) {
            result_file<<C_naive[i * p + j]<<" ";
        }

    }
    // Measure tiled CUDA performance
    // TODO: Launch tiled_cuda_matmul kernel
    cudaMemset(d_C, 0, m * p * sizeof(float));  // Reuse memory
    cudaEventRecord(start_tiled);
    tiled_cuda_matmul<<<dimGrid, dimBlock>>>(d_C, d_A, d_B, m, n, p, TILE_WIDTH);
    cudaDeviceSynchronize();

    cudaMemcpy(C_tiled, d_C, m * p * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop_tiled);
    cudaEventSynchronize(stop_tiled);
    float tiled_cuda_time = 0;
    cudaEventElapsedTime(&tiled_cuda_time, start_tiled, stop_tiled);
    tiled_cuda_time = tiled_cuda_time / 1000;

    // TODO: Write tiled CUDA result to file and validate
    std::ofstream result_file2(folder + "result_tiled.raw");

    result_file2 << m << " " << p << std::endl;
    for (int i = 0; i < m;result_file2<<std::endl, i++) {
        for (int j = 0; j < p; j++) {
            result_file2<<C_tiled[i * p + j]<<" ";
        }

    }

    //validation of the results
    bool validnative = validate_result(folder + "result_naive.raw", folder + "output.raw");
    if (!validnative) {
        std::cerr << "Naive result validation failed for case " << case_number << std::endl;
    }

    bool validtiled = validate_result(folder + "result_tiled.raw", folder + "output.raw");
    if (!validtiled) {
        std::cerr << "Tiled result validation failed for case " << case_number << std::endl;
    }
    // Print performance results
    std::cout << "Case " << case_number << " (" << m << "x" << n << "x" << p << "):\n";
    std::cout << "Naive CUDA time: " << naive_cuda_time << " seconds\n";
    std::cout << "Tiled CUDA time: " << tiled_cuda_time << " seconds\n";

    // Clean up

    cudaEventDestroy(start_naive);
    cudaEventDestroy(stop_naive);
    cudaEventDestroy(start_tiled);
    cudaEventDestroy(stop_tiled);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}