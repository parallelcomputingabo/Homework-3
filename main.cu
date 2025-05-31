#include <iostream>
#include <fstream>
#include <string>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

__global__ void naive_cuda_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p) {
    // Calculate global thread indices
    uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check bounds
    if (row < m && col < p) {
        float sum = 0.0f;
        for (uint32_t k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * p + col];
        }
        C[row * p + col] = sum;
    }
}

__global__ void tiled_cuda_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p, uint32_t tile_width) {
    // Shared memory tiles
    extern __shared__ float shared_mem[];
    float *tile_A = shared_mem;
    float *tile_B = shared_mem + tile_width * tile_width;
    
    // Calculate global indices
    uint32_t row = blockIdx.y * tile_width + threadIdx.y;
    uint32_t col = blockIdx.x * tile_width + threadIdx.x;
    
    float sum = 0.0f;
    
    // Loop through tiles
    for (uint32_t tile = 0; tile < (n + tile_width - 1) / tile_width; ++tile) {
        // Load tile from A into shared memory
        uint32_t a_col = tile * tile_width + threadIdx.x;
        if (row < m && a_col < n) {
            tile_A[threadIdx.y * tile_width + threadIdx.x] = A[row * n + a_col];
        } else {
            tile_A[threadIdx.y * tile_width + threadIdx.x] = 0.0f;
        }
        
        // Load tile from B into shared memory
        uint32_t b_row = tile * tile_width + threadIdx.y;
        if (b_row < n && col < p) {
            tile_B[threadIdx.y * tile_width + threadIdx.x] = B[b_row * p + col];
        } else {
            tile_B[threadIdx.y * tile_width + threadIdx.x] = 0.0f;
        }
        
        // Synchronize threads
        __syncthreads();
        
        // Compute partial dot product
        for (uint32_t k = 0; k < tile_width; ++k) {
            sum += tile_A[threadIdx.y * tile_width + k] * tile_B[k * tile_width + threadIdx.x];
        }
        
        // Synchronize before loading next tile
        __syncthreads();
    }
    
    // Write result
    if (row < m && col < p) {
        C[row * p + col] = sum;
    }
}

bool validate_result(const std::string &result_file, const std::string &reference_file, float epsilon = 1e-5) {
    std::ifstream result(result_file), reference(reference_file);
    if (!result || !reference) {
        std::cerr << "Validation failed: could not open files.\n";
        return false;
    }

    uint32_t r1, c1, r2, c2;
    result >> r1 >> c1;
    reference >> r2 >> c2;

    if (r1 != r2 || c1 != c2) {
        std::cerr << "Dimension mismatch during validation.\n";
        return false;
    }

    float a, b;
    for (uint32_t i = 0; i < r1 * c1; ++i) {
        result >> a;
        reference >> b;
        if (std::fabs(a - b) > epsilon) {
            std::cerr << "Mismatch at index " << i << ": " << a << " vs " << b << "\n";
            return false;
        }
    }

    return true;
}

bool read_matrix(const std::string &path, float *&M, uint32_t &rows, uint32_t &cols) {
    std::ifstream in(path);
    if (!in)
        return false;
    in >> rows >> cols;
    M = new float[static_cast<size_t>(rows) * cols];
    for (size_t i = 0; i < (size_t)rows * cols; ++i) {
        in >> M[i];
    }
    return true;
}

bool write_matrix(const std::string &path, float *M, uint32_t rows, uint32_t cols) {
    std::ofstream out(path);
    if (!out)
        return false;
    out << rows << " " << cols << '\n';
    for (size_t i = 0; i < (size_t)rows * cols; ++i) {
        out << M[i] << ((i + 1) % cols ? ' ' : '\n');
    }
    return true;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <case_number>" << std::endl;
        return 1;
    }

    int case_number = std::atoi(argv[1]);
    if (case_number < 0 || case_number > 9) {
        std::cerr << "Case number must be between 0 and 9" << std::endl;
        return 1;
    }

    // Construct file paths
    std::string folder = "data/" + std::to_string(case_number) + "/";
    std::string input0_file = folder + "input0.raw";
    std::string input1_file = folder + "input1.raw";
    std::string result_file = folder + "result.raw";
    std::string reference_file = folder + "output.raw";

    float *A = nullptr, *B = nullptr;
    uint32_t m, n, p;

    // Read input matrices
    if (!read_matrix(input0_file, A, m, n) || !read_matrix(input1_file, B, n, p)) {
        std::cerr << "Error reading input files" << std::endl;
        return EXIT_FAILURE;
    }

    // Allocate host memory for results
    float *C_naive = new float[m * p];
    float *C_tiled = new float[m * p];

    // Allocate GPU memory
    float *d_A, *d_B, *d_C;
    size_t size_A = m * n * sizeof(float);
    size_t size_B = n * p * sizeof(float);
    size_t size_C = m * p * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));

    // Copy matrices to GPU
    CUDA_CHECK(cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice));

    // Define block and grid dimensions for naive implementation
    dim3 block_size(16, 16);
    dim3 grid_size((p + block_size.x - 1) / block_size.x, (m + block_size.y - 1) / block_size.y);

    // Measure naive CUDA performance
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    naive_cuda_matmul<<<grid_size, block_size>>>(d_C, d_A, d_B, m, n, p);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float naive_cuda_time;
    CUDA_CHECK(cudaEventElapsedTime(&naive_cuda_time, start, stop));
    naive_cuda_time /= 1000.0f; // Convert to seconds

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(C_naive, d_C, size_C, cudaMemcpyDeviceToHost));

    // Write naive CUDA result to file and validate
    if (!write_matrix(result_file, C_naive, m, p)) {
        std::cerr << "Error writing naive CUDA result" << std::endl;
        return EXIT_FAILURE;
    }

    bool naive_correct = validate_result(result_file, reference_file);
    if (!naive_correct) {
        std::cerr << "Naive CUDA result validation failed for case " << case_number << std::endl;
    }

    // Measure tiled CUDA performance
    const uint32_t tile_width = 16;
    dim3 tiled_block_size(tile_width, tile_width);
    dim3 tiled_grid_size((p + tile_width - 1) / tile_width, (m + tile_width - 1) / tile_width);
    
    // Shared memory size: 2 tiles of size tile_width x tile_width
    size_t shared_mem_size = 2 * tile_width * tile_width * sizeof(float);

    CUDA_CHECK(cudaEventRecord(start));
    tiled_cuda_matmul<<<tiled_grid_size, tiled_block_size, shared_mem_size>>>(d_C, d_A, d_B, m, n, p, tile_width);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float tiled_cuda_time;
    CUDA_CHECK(cudaEventElapsedTime(&tiled_cuda_time, start, stop));
    tiled_cuda_time /= 1000.0f; // Convert to seconds

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(C_tiled, d_C, size_C, cudaMemcpyDeviceToHost));

    // Write tiled CUDA result to file and validate
    if (!write_matrix(result_file, C_tiled, m, p)) {
        std::cerr << "Error writing tiled CUDA result" << std::endl;
        return EXIT_FAILURE;
    }

    bool tiled_correct = validate_result(result_file, reference_file);
    if (!tiled_correct) {
        std::cerr << "Tiled CUDA result validation failed for case " << case_number << std::endl;
    }

    // Print performance results
    std::cout << "Case " << case_number << " (" << m << "x" << n << "x" << p << "):\n";
    std::cout << "Naive CUDA time: " << naive_cuda_time << " seconds\n";
    std::cout << "Tiled CUDA time: " << tiled_cuda_time << " seconds\n";
    std::cout << "Tiled speedup over naive CUDA: " << (naive_cuda_time / tiled_cuda_time) << "x\n";

    // Clean up
    delete[] A;
    delete[] B;
    delete[] C_naive;
    delete[] C_tiled;
    
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}