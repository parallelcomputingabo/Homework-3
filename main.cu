#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include <iomanip>

// ----------------------------------------------------------------------------
//  Text-based Matrix File I/O: Load and save matrices in the format you provided
// ----------------------------------------------------------------------------

/**
 * Reads a matrix from a text file.
 * The first line of the file contains the dimensions (rows and columns).
 * The remaining lines contain all matrix elements in row-major order.
 * This function will read everything into a std::vector<float>.
 */
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

/**
 * Writes a matrix to a text file in the same format as input:
 * First line is dimensions, then matrix elements row by row, space-separated.
 */
bool write_matrix(const std::string& path, const std::vector<float>& data, uint32_t rows, uint32_t cols) {
    std::ofstream file(path);
    if (!file) {
        std::cerr << "Failed to write to " << path << std::endl;
        return false;
    }
    file << rows << " " << cols << "\n";
    file << std::fixed << std::setprecision(2);  // Formatting for better readability
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < cols; ++j) {
            file << data[i * cols + j] << " ";
        }
        file << "\n";
    }
    return true;
}

/**
 * Compares two matrices in text files for validation.
 * Checks each element and prints a message if there's a mismatch.
 */
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

// ----------------------------------------------------------------------------
//  CUDA Kernels: These run in parallel on the GPU
// ----------------------------------------------------------------------------

/**
 * Naive CUDA matrix multiplication kernel.
 * Each thread computes a single element of the output matrix C.
 * Only global memory is used, no tiling or shared memory optimizations.
 */
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

/**
 * Tiled CUDA matrix multiplication kernel.
 * Uses shared memory to load "tiles" of A and B for each block, reducing global memory traffic.
 * Each block computes a TILE_WIDTH x TILE_WIDTH chunk of C.
 */
#define TILE_WIDTH 16
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

        // Perform multiplication for this tile
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

// ----------------------------------------------------------------------------
//  Host-side CUDA Wrapper: Allocates memory, launches kernels, times everything
// ----------------------------------------------------------------------------

/**
 * Helper to run either the naive or tiled CUDA matrix multiplication.
 * Handles device memory allocation, copying data, kernel launch, result copy-back, and timing.
 * Returns the elapsed time in seconds (and milliseconds via reference parameter).
 */
float run_cuda_matmul(const float *A, const float *B, float *C, uint32_t m, uint32_t n, uint32_t p,
                     bool use_tiled, float &gpu_time_ms)
{
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    size_t sz_A = m * n * sizeof(float);
    size_t sz_B = n * p * sizeof(float);
    size_t sz_C = m * p * sizeof(float);

    // Allocate GPU memory for input and output matrices
    cudaMalloc(&d_A, sz_A);
    cudaMalloc(&d_B, sz_B);
    cudaMalloc(&d_C, sz_C);

    // Copy input matrices from host (CPU) to device (GPU)
    cudaMemcpy(d_A, A, sz_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sz_B, cudaMemcpyHostToDevice);

    // Define block and grid dimensions for launching kernel
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((p + TILE_WIDTH - 1) / TILE_WIDTH, (m + TILE_WIDTH - 1) / TILE_WIDTH);

    // CUDA events for precise timing of kernel execution
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start timing, launch the appropriate kernel, stop timing
    cudaEventRecord(start);
    if (use_tiled)
        tiled_cuda_matmul<<<gridDim, blockDim>>>(d_C, d_A, d_B, m, n, p);
    else
        naive_cuda_matmul<<<gridDim, blockDim>>>(d_C, d_A, d_B, m, n, p);
    cudaEventRecord(stop);

    // Copy result back from device (GPU) to host (CPU)
    cudaMemcpy(C, d_C, sz_C, cudaMemcpyDeviceToHost);

    // Wait for kernel to finish and calculate elapsed time in milliseconds
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time_ms, start, stop);

    // Clean up GPU memory and events
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return gpu_time_ms / 1000.0f; // seconds
}

// ----------------------------------------------------------------------------
//  Main Program: Reads inputs, runs CUDA kernels, writes/validates outputs
// ----------------------------------------------------------------------------

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <test_case_number (0-9)>" << std::endl;
        return 1;
    }

    // Construct file paths for input and output
    std::string case_num = argv[1];
    std::string folder = "data/" + case_num + "/";
    std::string inputA = folder + "input0.raw";
    std::string inputB = folder + "input1.raw";
    std::string outputPath = folder + "result.raw";
    std::string expectedPath = folder + "output.raw";

    // Load input matrices from text files
    std::vector<float> A, B, C_naive, C_tiled;
    uint32_t m, n, nB, p;
    if (!load_matrix(inputA, A, m, n)) return 1;
    if (!load_matrix(inputB, B, nB, p)) return 1;
    if (n != nB) {
        std::cerr << "Matrix dimensions incompatible for multiplication.\n";
        return 1;
    }

    C_naive.resize(m * p, 0.0f);  // Output for naive CUDA
    C_tiled.resize(m * p, 0.0f);  // Output for tiled CUDA

    // Run Naive CUDA implementation
    float naive_gpu_ms;
    run_cuda_matmul(A.data(), B.data(), C_naive.data(), m, n, p, false, naive_gpu_ms);
    write_matrix(folder + "result_naive_cuda.raw", C_naive, m, p);
    bool naive_ok = validate_result(folder + "result_naive_cuda.raw", expectedPath, m, p);

    // Run Tiled CUDA implementation
    float tiled_gpu_ms;
    run_cuda_matmul(A.data(), B.data(), C_tiled.data(), m, n, p, true, tiled_gpu_ms);
    write_matrix(folder + "result_tiled_cuda.raw", C_tiled, m, p);
    bool tiled_ok = validate_result(folder + "result_tiled_cuda.raw", expectedPath, m, p);

    // Print timing and validation summary for this test case
    std::cout << "Case " << case_num << " (" << m << "x" << n << "x" << p << "):\n";
    std::cout << "Naive CUDA time: " << naive_gpu_ms / 1000.0f << " seconds - " << (naive_ok ? "CORRECT" : "INCORRECT") << "\n";
    std::cout << "Tiled CUDA time: " << tiled_gpu_ms / 1000.0f << " seconds - " << (tiled_ok ? "CORRECT" : "INCORRECT") << "\n";
    if (naive_ok && tiled_ok && tiled_gpu_ms > 0)
        std::cout << "Tiled CUDA speedup: " << (naive_gpu_ms / tiled_gpu_ms) << "x\n";
    else
        std::cout << "Tiled CUDA speedup: N/A\n";

    return 0;
}
