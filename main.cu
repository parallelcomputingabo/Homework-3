#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <chrono>

// Naive CUDA kernel: each thread computes one C(i,j)
__global__ void naive_cuda_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p) {
    uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < p) {
        float sum = 0.0f;
        for (uint32_t k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * p + col];
        }
        C[row * p + col] = sum;
    }
}

// Tiled CUDA kernel: uses shared memory for A and B tiles
__global__ void tiled_cuda_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p, uint32_t tile_width) {
    extern __shared__ float shared_mem[];
    float* As = shared_mem;
    float* Bs = shared_mem + tile_width * tile_width;

    uint32_t row = blockIdx.y * tile_width + threadIdx.y;
    uint32_t col = blockIdx.x * tile_width + threadIdx.x;
    float sum = 0.0f;

    for (uint32_t t = 0; t < (n + tile_width - 1) / tile_width; ++t) {
        // Load tiles into shared memory
        uint32_t tiled_row = row;
        uint32_t tiled_col = t * tile_width + threadIdx.x;
        if (tiled_row < m && tiled_col < n)
            As[threadIdx.y * tile_width + threadIdx.x] = A[tiled_row * n + tiled_col];
        else
            As[threadIdx.y * tile_width + threadIdx.x] = 0.0f;

        tiled_row = t * tile_width + threadIdx.y;
        tiled_col = col;
        if (tiled_row < n && tiled_col < p)
            Bs[threadIdx.y * tile_width + threadIdx.x] = B[tiled_row * p + tiled_col];
        else
            Bs[threadIdx.y * tile_width + threadIdx.x] = 0.0f;

        __syncthreads();

        for (uint32_t k = 0; k < tile_width; ++k) {
            sum += As[threadIdx.y * tile_width + k] * Bs[k * tile_width + threadIdx.x];
        }
        __syncthreads();
    }
    if (row < m && col < p)
        C[row * p + col] = sum;
}

// Helper: Read raw float matrix from file
bool read_matrix(const std::string &filename, std::vector<float> &mat, size_t size) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) return false;
    in.read(reinterpret_cast<char*>(mat.data()), size * sizeof(float));
    return in.good();
}

// Helper: Write raw float matrix to file
bool write_matrix(const std::string &filename, const std::vector<float> &mat, size_t size) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) return false;
    out.write(reinterpret_cast<const char*>(mat.data()), size * sizeof(float));
    return out.good();
}

// Helper: Validate result against reference
bool validate_result(const std::string &result_file, const std::string &reference_file) {
    std::ifstream res(result_file, std::ios::binary);
    std::ifstream ref(reference_file, std::ios::binary);
    if (!res || !ref) return false;
    res.seekg(0, std::ios::end);
    ref.seekg(0, std::ios::end);
    size_t res_size = res.tellg();
    size_t ref_size = ref.tellg();
    if (res_size != ref_size) return false;
    res.seekg(0);
    ref.seekg(0);
    std::vector<char> res_buf(res_size), ref_buf(ref_size);
    res.read(res_buf.data(), res_size);
    ref.read(ref_buf.data(), ref_size);
    return std::memcmp(res_buf.data(), ref_buf.data(), res_size) == 0;
}

int main(int argc, char *argv[]) {
    // Example: ./main 0 1024 1024 1024
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <case_number> <m> <n> <p>\n";
        return 1;
    }
    int case_number = std::stoi(argv[1]);
    uint32_t m = std::stoul(argv[2]);
    uint32_t n = std::stoul(argv[3]);
    uint32_t p = std::stoul(argv[4]);
    uint32_t tile_width = 16; // You can experiment with 16, 32, etc.

    std::string case_dir = "data/" + std::to_string(case_number) + "/";
    std::string a_file = case_dir + "input0.raw";
    std::string b_file = case_dir + "input1.raw";
    std::string c_file_naive = case_dir + "result_naive.raw";
    std::string c_file_tiled = case_dir + "result_tiled.raw";
    std::string ref_file = case_dir + "output.raw";

    size_t size_A = m * n;
    size_t size_B = n * p;
    size_t size_C = m * p;

    std::vector<float> h_A(size_A), h_B(size_B), h_C_naive(size_C), h_C_tiled(size_C);

    // Read input matrices
    if (!read_matrix(a_file, h_A, size_A) || !read_matrix(b_file, h_B, size_B)) {
        std::cerr << "Failed to read input files.\n";
        return 1;
    }

    // Allocate device memory
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    cudaMalloc(&d_A, size_A * sizeof(float));
    cudaMalloc(&d_B, size_B * sizeof(float));
    cudaMalloc(&d_C, size_C * sizeof(float));

    // --- Naive CUDA ---
    cudaEvent_t start_naive, stop_naive;
    cudaEventCreate(&start_naive);
    cudaEventCreate(&stop_naive);

    cudaEventRecord(start_naive);
    cudaMemcpy(d_A, h_A.data(), size_A * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size_B * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((p + 15) / 16, (m + 15) / 16);
    naive_cuda_matmul<<<numBlocks, threadsPerBlock>>>(d_C, d_A, d_B, m, n, p);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C_naive.data(), d_C, size_C * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop_naive);
    cudaEventSynchronize(stop_naive);

    float naive_cuda_time = 0.0f;
    cudaEventElapsedTime(&naive_cuda_time, start_naive, stop_naive);
    naive_cuda_time /= 1000.0f; // ms to s

    write_matrix(c_file_naive, h_C_naive, size_C);
    bool valid_naive = validate_result(c_file_naive, ref_file);

    // --- Tiled CUDA ---
    cudaEvent_t start_tiled, stop_tiled;
    cudaEventCreate(&start_tiled);
    cudaEventCreate(&stop_tiled);

    cudaEventRecord(start_tiled);
    cudaMemcpy(d_A, h_A.data(), size_A * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size_B * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlockTiled(tile_width, tile_width);
    dim3 numBlocksTiled((p + tile_width - 1) / tile_width, (m + tile_width - 1) / tile_width);
    size_t shared_mem_size = 2 * tile_width * tile_width * sizeof(float);
    tiled_cuda_matmul<<<numBlocksTiled, threadsPerBlockTiled, shared_mem_size>>>(d_C, d_A, d_B, m, n, p, tile_width);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C_tiled.data(), d_C, size_C * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop_tiled);
    cudaEventSynchronize(stop_tiled);

    float tiled_cuda_time = 0.0f;
    cudaEventElapsedTime(&tiled_cuda_time, start_tiled, stop_tiled);
    tiled_cuda_time /= 1000.0f; // ms to s

    write_matrix(c_file_tiled, h_C_tiled, size_C);
    bool valid_tiled = validate_result(c_file_tiled, ref_file);

    // Print performance results
    std::cout << "Case " << case_number << " (" << m << "x" << n << "x" << p << "):\n";
    std::cout << "Naive CUDA time: " << naive_cuda_time << " seconds";
    std::cout << (valid_naive ? " [VALID]\n" : " [INVALID]\n");
    std::cout << "Tiled CUDA time: " << tiled_cuda_time << " seconds";
    std::cout << (valid_tiled ? " [VALID]\n" : " [INVALID]\n");

    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start_naive);
    cudaEventDestroy(stop_naive);
    cudaEventDestroy(start_tiled);
    cudaEventDestroy(stop_tiled);

    return 0;
}