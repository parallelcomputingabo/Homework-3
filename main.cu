#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include <chrono>
#include <filesystem>

#define TILE_WIDTH 16

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

__global__ void tiled_cuda_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p, uint32_t tile_width) {
    __shared__ float tile_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tile_B[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * tile_width + threadIdx.y;
    int col = blockIdx.x * tile_width + threadIdx.x;
    float value = 0.0f;

    for (int ph = 0; ph < (n + tile_width - 1) / tile_width; ++ph) {
        if (row < m && ph * tile_width + threadIdx.x < n)
            tile_A[threadIdx.y][threadIdx.x] = A[row * n + ph * tile_width + threadIdx.x];
        else
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < p && ph * tile_width + threadIdx.y < n)
            tile_B[threadIdx.y][threadIdx.x] = B[(ph * tile_width + threadIdx.y) * p + col];
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

void read_matrix(const std::string &path, std::vector<float> &matrix, size_t size) {
    std::ifstream file(path, std::ios::binary);
    matrix.resize(size);
    file.read(reinterpret_cast<char *>(matrix.data()), size * sizeof(float));
}

void write_matrix(const std::string &path, const std::vector<float> &matrix) {
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char *>(matrix.data()), matrix.size() * sizeof(float));
}

bool validate_result(const std::string &result_path, const std::string &ref_path, size_t size) {
    std::vector<float> result(size), ref(size);
    read_matrix(result_path, result, size);
    read_matrix(ref_path, ref, size);
    for (size_t i = 0; i < size; ++i) {
        if (fabs(result[i] - ref[i]) > 1e-3f) return false;
    }
    return true;
}

int main() {
    const std::string base_dir = "E:/Abo Akademi University-Master Program/First Academic Year 2024-2025/4. Fourth Period/Paralell Computing/Homework-3-main/data/";

    for (int case_number = 0; case_number <= 9; ++case_number) {
        std::string base = base_dir + std::to_string(case_number) + "/";
        std::ifstream meta(base + "meta.txt");
        uint32_t m, n, p;
        meta >> m >> n >> p;

        size_t size_A = m * n, size_B = n * p, size_C = m * p;
        std::vector<float> A, B, C(size_C);
        read_matrix(base + "input0.raw", A, size_A);
        read_matrix(base + "input1.raw", B, size_B);

        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, size_A * sizeof(float));
        cudaMalloc(&d_B, size_B * sizeof(float));
        cudaMalloc(&d_C, size_C * sizeof(float));

        cudaMemcpy(d_A, A.data(), size_A * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B.data(), size_B * sizeof(float), cudaMemcpyHostToDevice);

        dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
        dim3 dimGrid((p + TILE_WIDTH - 1) / TILE_WIDTH, (m + TILE_WIDTH - 1) / TILE_WIDTH);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Naive
        cudaEventRecord(start);
        naive_cuda_matmul<<<dimGrid, dimBlock>>>(d_C, d_A, d_B, m, n, p);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float naive_ms = 0;
        cudaEventElapsedTime(&naive_ms, start, stop);

        cudaMemcpy(C.data(), d_C, size_C * sizeof(float), cudaMemcpyDeviceToHost);
        write_matrix(base + "naive_result.raw", C);
        bool naive_valid = validate_result(base + "naive_result.raw", base + "output.raw", size_C);

        // Tiled
        cudaEventRecord(start);
        tiled_cuda_matmul<<<dimGrid, dimBlock>>>(d_C, d_A, d_B, m, n, p, TILE_WIDTH);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float tiled_ms = 0;
        cudaEventElapsedTime(&tiled_ms, start, stop);

        cudaMemcpy(C.data(), d_C, size_C * sizeof(float), cudaMemcpyDeviceToHost);
        write_matrix(base + "tiled_result.raw", C);
        bool tiled_valid = validate_result(base + "tiled_result.raw", base + "output.raw", size_C);

        printf("Case %d (%dx%dx%d):\n", case_number, m, n, p);
        printf("Naive CUDA time: %.4f s [%s]\n", naive_ms / 1000.0f, naive_valid ? "OK" : "FAIL");
        printf("Tiled CUDA time: %.4f s [%s]\n\n", tiled_ms / 1000.0f, tiled_valid ? "OK" : "FAIL");

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    return 0;
}
