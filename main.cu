#include <cuda_runtime.h>
#include <cassert>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>

// Error checking macro
#define CUDA_CHECK(err) do { \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// Naive CUDA Matrix Multiplication Kernel
__global__ void naive_cuda_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < p) {
        float sum = 0.0f;
        for (uint32_t k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * p + col];
        }
        C[row * p + col] = sum;
    }
}

// Tiled CUDA Matrix Multiplication Kernel
__global__ void tiled_cuda_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p, uint32_t tile_width) {
    __shared__ float As[32][32]; // TILE_WIDTH=32
    __shared__ float Bs[32][32];

    int row = blockIdx.y * tile_width + threadIdx.y;
    int col = blockIdx.x * tile_width + threadIdx.x;
    float sum = 0.0f;

    for (uint32_t t = 0; t < (n + tile_width - 1) / tile_width; ++t) {
        // Load tiles into shared memory
        if (row < m && (t * tile_width + threadIdx.x) < n) {
            As[threadIdx.y][threadIdx.x] = A[row * n + t * tile_width + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        if (col < p && (t * tile_width + threadIdx.y) < n) {
            Bs[threadIdx.y][threadIdx.x] = B[(t * tile_width + threadIdx.y) * p + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();

        // Compute partial sum
        for (uint32_t k = 0; k < tile_width; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < m && col < p) {
        C[row * p + col] = sum;
    }
}

// Function to read matrix from file
void read_matrix(const std::string& filename, std::vector<float>& matrix, uint32_t rows, uint32_t cols) {
    matrix.resize(rows * cols);
    std::ifstream file(filename, std::ios::binary);
    assert(file.is_open());
    file.read(reinterpret_cast<char*>(matrix.data()), rows * cols * sizeof(float));
    file.close();
}

// Function to write matrix to file
void write_matrix(const std::string& filename, const std::vector<float>& matrix) {
    std::ofstream file(filename, std::ios::binary);
    assert(file.is_open());
    file.write(reinterpret_cast<const char*>(matrix.data()), matrix.size() * sizeof(float));
    file.close();
}

// Function to run matrix multiplication and measure time
float run_matrix_multiplication(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C,
                               uint32_t m, uint32_t n, uint32_t p, bool tiled, uint32_t tile_width = 32) {
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, m * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, n * p * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, m * p * sizeof(float)));

    // Transfer data to device
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), m * n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), n * p * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 threadsPerBlock(tiled ? tile_width : 16, tiled ? tile_width : 16);
    dim3 blocksPerGrid((p + threadsPerBlock.x - 1) / threadsPerBlock.x, (m + threadsPerBlock.y - 1) / threadsPerBlock.y);
    if (tiled) {
        tiled_cuda_matmul<<<blocksPerGrid, threadsPerBlock>>>(d_C, d_A, d_B, m, n, p, tile_width);
    } else {
        naive_cuda_matmul<<<blocksPerGrid, threadsPerBlock>>>(d_C, d_A, d_B, m, n, p);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Transfer result back to host
    CUDA_CHECK(cudaMemcpy(C.data(), d_C, m * p * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return milliseconds / 1000.0f; // Convert to seconds
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <test_case>" << std::endl;
        return 1;
    }

    std::string test_case = argv[1];
    std::string data_path = "data/" + test_case + "/";
    uint32_t m, n, p;

    // Read dimensions
    std::ifstream dim_file(data_path + "dim.txt");
    assert(dim_file.is_open());
    dim_file >> m >> n >> p;
    dim_file.close();

    // Read input matrices
    std::vector<float> A, B, C(m * p);
    read_matrix(data_path + "input0.raw", A, m, n);
    read_matrix(data_path + "input1.raw", B, n, p);

    // Run and time naive CUDA
    float naive_time = run_matrix_multiplication(A, B, C, m, n, p, false);
    write_matrix(data_path + "result_naive.raw", C);

    // Run and time tiled CUDA
    float tiled_time = run_matrix_multiplication(A, B, C, m, n, p, true, 32);
    write_matrix(data_path + "result_tiled.raw", C);

    // Output performance results
    std::cout << "Test Case: " << test_case << std::endl;
    std::cout << "Dimensions (m x n x p): " << m << " x " << n << " x " << p << std::endl;
    std::cout << "Naive CUDA Time: " << std::fixed << std::setprecision(6) << naive_time << " s" << std::endl;
    std::cout << "Tiled CUDA Time: " << tiled_time << " s" << std::endl;
    std::cout << "Tiled CUDA Speedup (vs. Naive CUDA): " << naive_time / tiled_time << "x" << std::endl;

    return 0;
}