#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include <iomanip>

// CUDA error checking macro
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void naive_cuda_matmul(float *C, const float *A, const float *B, uint32_t m, uint32_t n, uint32_t p) {
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

#define TILE_WIDTH 32
__global__ void tiled_cuda_matmul(float *C, const float *A, const float *B, uint32_t m, uint32_t n, uint32_t p) {
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    float value = 0.0f;

    // Loop over tiles required to compute output element
    for (uint32_t t = 0; t < (n + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
        // Each thread loads one element of A and B into shared memory if in bounds, otherwise zero
        if (row < m && t * TILE_WIDTH + threadIdx.x < n)
            As[threadIdx.y][threadIdx.x] = A[row * n + t * TILE_WIDTH + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < p && t * TILE_WIDTH + threadIdx.y < n)
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_WIDTH + threadIdx.y) * p + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();  // Wait for all threads in the block to finish loading

         // Compute partial result for this tile
        for (uint32_t k = 0; k < TILE_WIDTH; ++k) {
            value += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();  // Make sure computation is done before loading next tile
    }

    // Write the computed value to C if within matrix bounds
    if (row < m && col < p) {
        C[row * p + col] = value;
    }
}

// Read matrix from .raw file
bool load_matrix(const std::string& path, std::vector<float>& data, uint32_t& rows, uint32_t& cols) {
    std::ifstream file(path);
    if (!file) {
        std::cerr << "Failed to open " << path << std::endl;
        return false;
    }
    file >> rows >> cols;  // Read the first line (matrix dimensions)
    data.resize(rows * cols);
    for (uint32_t i = 0; i < rows * cols; ++i) {
        file >> data[i];  // Read all matrix elements as floats
    }
    return true;
}

// Write matrix to .raw file
bool write_matrix(const std::string& path, const std::vector<float>& data, uint32_t rows, uint32_t cols) {
    std::ofstream file(path);
    if (!file) {
        std::cerr << "Failed to write to " << path << std::endl;
        return false;
    }
    file << rows << " " << cols << "\n";
    file << std::fixed << std::setprecision(2);
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < cols; ++j) {
            file << data[i * cols + j] << " ";
        }
        file << "\n";
    }
    return true;
}

// Validate result against reference
bool validate_result(const std::string& result_file, const std::string& reference_file, uint32_t rows, uint32_t cols) {
    std::ifstream res(result_file), ref(reference_file);
    if (!res || !ref) {
        std::cerr << "Failed to open files for validation.\n";
        return false;
    }
    uint32_t r_rows, r_cols, ref_rows, ref_cols;
    res >> r_rows >> r_cols;
    ref >> ref_rows >> ref_cols;
    if (r_rows != ref_rows || r_cols != ref_cols) {
        std::cerr << "Dimension mismatch in validation.\n";
        return false;
    }
    for (size_t i = 0; i < size_t(r_rows) * r_cols; ++i) {
        float a, b;
        res >> a;
        ref >> b;
        if (std::abs(a - b) > 1e-2) {
            std::cerr << "Mismatch at index " << i << ": " << a << " vs " << b << std::endl;
            return false;
        }
    }
    return true;
}


// Run CUDA matrix multiplication and measure time
float run_cuda_matmul(const float *A, const float *B, float *C, uint32_t m, uint32_t n, uint32_t p,
                     bool use_tiled, float &gpu_time_ms, int iterations = 5)
{
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    size_t sz_A = m * n * sizeof(float);
    size_t sz_B = n * p * sizeof(float);
    size_t sz_C = m * p * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_A, sz_A));
    CUDA_CHECK(cudaMalloc(&d_B, sz_B));
    CUDA_CHECK(cudaMalloc(&d_C, sz_C));

    CUDA_CHECK(cudaMemcpy(d_A, A, sz_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, sz_B, cudaMemcpyHostToDevice));

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((p + TILE_WIDTH - 1) / TILE_WIDTH, (m + TILE_WIDTH - 1) / TILE_WIDTH);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    float total_ms = 0.0f;
    for (int i = 0; i < iterations; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        if (use_tiled)
            tiled_cuda_matmul<<<gridDim, blockDim>>>(d_C, d_A, d_B, m, n, p);
        else
            naive_cuda_matmul<<<gridDim, blockDim>>>(d_C, d_A, d_B, m, n, p);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        total_ms += ms;
    }

    CUDA_CHECK(cudaMemcpy(C, d_C, sz_C, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    gpu_time_ms = total_ms / iterations;
    return gpu_time_ms / 1000.0f; // seconds
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <case number>" << std::endl;
        return 1;
    }

    // Construct file paths for input and output
    std::string case_num = argv[1];
    std::string folder = "data/" + case_num + "/";
    std::string input0_file = folder + "input0.raw";
    std::string input1_file = folder + "input1.raw";
    std::string result_file = folder + "result.raw";
    std::string reference_file = folder + "output.raw";

    // Load input matrices from text files
    std::vector<float> A, B, C_naive, C_tiled;
    uint32_t m, n, nB, p;
    if (!load_matrix(input0_file, A, m, n)) return 1;
    if (!load_matrix(input1_file, B, nB, p)) return 1;
    if (n != nB) {
        std::cerr << "Matrix dimensions incompatible for multiplication.\n";
        return 1;
    }

    C_naive.resize(m * p, 0.0f);  // Output for naive CUDA
    C_tiled.resize(m * p, 0.0f);  // Output for tiled CUDA

    // Run Naive CUDA implementation
    float naive_cuda_time;
    run_cuda_matmul(A.data(), B.data(), C_naive.data(), m, n, p, false, naive_cuda_time, 5);
    write_matrix(result_file, C_naive, m, p);
    bool valid_naive = validate_result(result_file, reference_file, m, p);
    
    // check if the naive result is valid
    if (!valid_naive) {
        std::cerr << "Naive CUDA result validation failed.\n";
        return 1; // Exit early if validation fails
    }

    // Run Tiled CUDA implementation
    float tiled_cuda_time;
    run_cuda_matmul(A.data(), B.data(), C_tiled.data(), m, n, p, true, tiled_cuda_time, 5);
    write_matrix(result_file, C_tiled, m, p);
    bool valid_tiled = validate_result(result_file, reference_file, m, p);

    // check if the tiled result is valid
    if (!valid_tiled) {
        std::cerr << "Tiled CUDA result validation failed.\n";
        return 1; // Exit early if validation fails
    }

    // Print timing and validation summary for this test case
    std::cout << "Case " << case_num << " (" << m << "x" << n << "x" << p << "):\n";
     std::cout << "Naive CUDA time: " << naive_cuda_time << " seconds. ";
    std::cout << (valid_naive ? "PASS" : "FAIL") << std::endl;
    std::cout << "Tiled CUDA time: " << tiled_cuda_time << " seconds. ";
    std::cout << (valid_tiled ? "PASS" : "FAIL") << std::endl;
    if (valid_naive && valid_tiled && tiled_cuda_time > 0)
        std::cout << "Tiled CUDA speedup: " << (naive_cuda_time / tiled_cuda_time) << "x\n";
    else
        std::cout << "Tiled CUDA speedup: N/A\n";

    return 0;
}