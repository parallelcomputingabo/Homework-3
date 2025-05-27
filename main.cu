#include <iostream>
#include <fstream>
#include <string>
#include <cuda_runtime.h>
#include <chrono>
#include <limits>  // For numeric_limits

// For debugging, reduce MAX_TILE_WIDTH
#define MAX_TILE_WIDTH 16

// Naive CUDA matrix multiplication kernel
__global__ void naive_cuda_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p) {
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

// Tiled CUDA matrix multiplication kernel using dynamic shared memory
__global__ void tiled_cuda_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p, uint32_t tile_width) {
    // Use dynamic shared memory
    extern __shared__ float shared_mem[];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * tile_width + ty;
    int col = bx * tile_width + tx;
    
    float sum = 0.0f;
    
    // Calculate shared memory offsets
    float* ds_A = shared_mem;
    float* ds_B = shared_mem + tile_width * tile_width;
    
    // Loop over tiles
    for (int t = 0; t < (n + tile_width - 1) / tile_width; t++) {
        // Collaborative loading of tiles into shared memory
        // Load tile of A
        if (row < m && t * tile_width + tx < n) {
            ds_A[ty * tile_width + tx] = A[row * n + t * tile_width + tx];
        } else {
            ds_A[ty * tile_width + tx] = 0.0f;
        }
        
        // Load tile of B
        if (col < p && t * tile_width + ty < n) {
            ds_B[ty * tile_width + tx] = B[(t * tile_width + ty) * p + col];
        } else {
            ds_B[ty * tile_width + tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        for (int i = 0; i < tile_width; i++) {
            sum += ds_A[ty * tile_width + i] * ds_B[i * tile_width + tx];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < m && col < p) {
        C[row * p + col] = sum;
    }
}

bool validate_result(const std::string &result_file, const std::string &reference_file) {
    std::ifstream result(result_file);
    std::ifstream reference(reference_file);
    
    if (!result || !reference) {
        std::cerr << "Error opening files for validation" << std::endl;
        return false;
    }
    
    uint32_t m1, p1, m2, p2;
    result >> m1 >> p1;
    reference >> m2 >> p2;
    
    if (m1 != m2 || p1 != p2) {
        std::cerr << "Matrix dimensions don't match between result and reference" << std::endl;
        return false;
    }
    
    float r1, r2;
    for (uint32_t i = 0; i < m1 * p1; i++) {
        if (!(result >> r1) || !(reference >> r2)) {
            std::cerr << "Error reading matrix elements" << std::endl;
            return false;
        }
        // Use relative tolerance for better floating-point comparison
        float abs_diff = std::abs(r1 - r2);
        float rel_diff = abs_diff / (std::abs(r2) + 1e-10f); // Add small epsilon to avoid division by zero
        
        if (abs_diff > 1e-2f && rel_diff > 1e-4f) { // More lenient tolerance
            std::cerr << "Matrix element mismatch at position " << i << ": " 
                      << r1 << " vs " << r2 << " (abs diff: " << abs_diff 
                      << ", rel diff: " << rel_diff << ")" << std::endl;
            return false;
        }
    }
    
    return true;
}

// For testing, let's create a simple matrix test function
void test_cuda_matrix_multiplication() {
    const int size = 4; // Small test case
    float h_A[size*size] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    float h_B[size*size] = {16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
    float h_C[size*size] = {0};
    
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size * size * sizeof(float));
    cudaMalloc(&d_B, size * size * sizeof(float));
    cudaMalloc(&d_C, size * size * sizeof(float));
    
    cudaMemcpy(d_A, h_A, size * size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size * size * sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 blockDim(2, 2);
    dim3 gridDim(2, 2);
    
    std::cout << "Running simple test with 4x4 matrices..." << std::endl;
    naive_cuda_matmul<<<gridDim, blockDim>>>(d_C, d_A, d_B, size, size, size);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_C, d_C, size * size * sizeof(float), cudaMemcpyDeviceToHost);
    
    std::cout << "Result matrix:" << std::endl;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            std::cout << h_C[i * size + j] << " ";
        }
        std::cout << std::endl;
    }
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <case_number>" << std::endl;
        std::cerr << "       " << argv[0] << " test (for simple test)" << std::endl;
        return 1;
    }
    
    std::string arg(argv[1]);
    if (arg == "test") {
        test_cuda_matrix_multiplication();
        return 0;
    }
    
    int case_number = std::stoi(argv[1]);
    std::string case_dir = "data/" + std::to_string(case_number) + "/";
    
    std::cout << "Processing case " << case_number << " in directory: " << case_dir << std::endl;
    
    // Read matrix dimensions and data from text files
    std::ifstream input0(case_dir + "input0.raw");
    std::ifstream input1(case_dir + "input1.raw");
    
    if (!input0 || !input1) {
        std::cerr << "Error: Could not open input files" << std::endl;
        return 1;
    }
    
    uint32_t m, n, p;
    input0 >> m >> n;  // Read dimensions of first matrix
    input1 >> n >> p;  // Read dimensions of second matrix
    
    std::cout << "Matrix dimensions: A(" << m << "x" << n << "), B(" << n << "x" << p << ")" << std::endl;
    
    // Check if dimensions make sense
    if (m <= 0 || n <= 0 || p <= 0 || m > 1000 || n > 1000 || p > 1000) {
        std::cerr << "Invalid matrix dimensions: A(" << m << "x" << n << "), B(" << n << "x" << p << ")" << std::endl;
        return 1;
    }
    
    // Skip any remaining characters on the line
    input0.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    input1.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    
    // Allocate host memory
    float *h_A = nullptr;
    float *h_B = nullptr;
    float *h_C_naive = nullptr;
    float *h_C_tiled = nullptr;
    
    try {
        h_A = new float[m * n];
        h_B = new float[n * p];
        h_C_naive = new float[m * p];
        h_C_tiled = new float[m * p];
    } catch (std::bad_alloc& e) {
        std::cerr << "Memory allocation failed: " << e.what() << std::endl;
        return 1;
    }
    
    // Initialize arrays to zero
    for (uint32_t i = 0; i < m * n; i++) h_A[i] = 0.0f;
    for (uint32_t i = 0; i < n * p; i++) h_B[i] = 0.0f;
    for (uint32_t i = 0; i < m * p; i++) h_C_naive[i] = h_C_tiled[i] = 0.0f;
    
    // Read matrix data, handling line breaks
    std::cout << "Reading matrix A..." << std::endl;
    for (uint32_t i = 0; i < m * n; i++) {
        if (!(input0 >> h_A[i])) {
            std::cerr << "Error reading matrix A at element " << i << std::endl;
            return 1;
        }
    }
    
    std::cout << "Reading matrix B..." << std::endl;
    for (uint32_t i = 0; i < n * p; i++) {
        if (!(input1 >> h_B[i])) {
            std::cerr << "Error reading matrix B at element " << i << std::endl;
            return 1;
        }
    }
    
    // Print a sample of matrices to verify data
    std::cout << "Matrix A (first few elements): ";
    for (int i = 0; i < 5 && i < m * n; i++) {
        std::cout << h_A[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Matrix B (first few elements): ";
    for (int i = 0; i < 5 && i < n * p; i++) {
        std::cout << h_B[i] << " ";
    }
    std::cout << std::endl;
    
    // Close input files
    input0.close();
    input1.close();
    
    // Allocate device memory
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    cudaMalloc(&d_A, m * n * sizeof(float));
    cudaMalloc(&d_B, n * p * sizeof(float));
    cudaMalloc(&d_C, m * p * sizeof(float));
    
    // Copy input data to device
    cudaMemcpy(d_A, h_A, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n * p * sizeof(float), cudaMemcpyHostToDevice);
    
    // Define block and grid dimensions
    // Choose 16 as our operational tile width
    const uint32_t tile_width = 16;
    dim3 blockDim(tile_width, tile_width);
    dim3 gridDim((p + blockDim.x - 1) / blockDim.x, (m + blockDim.y - 1) / blockDim.y);
    
    std::cout << "Grid dimensions: (" << gridDim.x << "x" << gridDim.y << "), Block dimensions: (" 
              << blockDim.x << "x" << blockDim.y << ")" << std::endl;
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Measure naive CUDA performance
    std::cout << "Running naive CUDA kernel..." << std::endl;
    cudaEventRecord(start);
    naive_cuda_matmul<<<gridDim, blockDim>>>(d_C, d_A, d_B, m, n, p);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float naive_cuda_time;
    cudaEventElapsedTime(&naive_cuda_time, start, stop);
    naive_cuda_time /= 1000.0f; // Convert to seconds
    
    // Copy result back to host
    cudaMemcpy(h_C_naive, d_C, m * p * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Write naive result to file
    std::ofstream naive_result(case_dir + "result_naive.raw");
    if (!naive_result) {
        std::cerr << "Error opening file for writing: " << case_dir + "result_naive.raw" << std::endl;
        return 1;
    }
    naive_result << m << " " << p << "\n";  // Write dimensions
    for (uint32_t i = 0; i < m * p; i++) {
        naive_result << h_C_naive[i] << " ";
    }
    naive_result << "\n";
    naive_result.close();
    std::cout << "Naive CUDA result written to: " << case_dir + "result_naive.raw" << std::endl;
    
    // Measure tiled CUDA performance
    std::cout << "Running tiled CUDA kernel with tile width: " << tile_width << "..." << std::endl;
    cudaEventRecord(start);
    // Need to allocate 2 * tile_width * tile_width floats of shared memory (for both matrices A and B)
    tiled_cuda_matmul<<<gridDim, blockDim, 2 * tile_width * tile_width * sizeof(float)>>>(d_C, d_A, d_B, m, n, p, tile_width);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float tiled_cuda_time;
    cudaEventElapsedTime(&tiled_cuda_time, start, stop);
    tiled_cuda_time /= 1000.0f; // Convert to seconds
    
    // Copy result back to host
    cudaMemcpy(h_C_tiled, d_C, m * p * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Write tiled result to file
    std::ofstream tiled_result(case_dir + "result_tiled.raw");
    if (!tiled_result) {
        std::cerr << "Error opening file for writing: " << case_dir + "result_tiled.raw" << std::endl;
        return 1;
    }
    tiled_result << m << " " << p << "\n";  // Write dimensions
    for (uint32_t i = 0; i < m * p; i++) {
        tiled_result << h_C_tiled[i] << " ";
    }
    tiled_result << "\n";
    tiled_result.close();
    std::cout << "Tiled CUDA result written to: " << case_dir + "result_tiled.raw" << std::endl;

    // Print performance results
    std::cout << "\nCase " << case_number << " (" << m << "x" << n << "x" << p << "):\n";
    std::cout << "Naive CUDA time: " << naive_cuda_time << " seconds\n";
    std::cout << "Tiled CUDA time: " << tiled_cuda_time << " seconds\n";
    std::cout << "Speedup of tiled vs naive: " << (naive_cuda_time / tiled_cuda_time) << "x\n";
    
    // Validate results
    std::cout << "\nValidating results..." << std::endl;
    bool naive_valid = validate_result(case_dir + "result_naive.raw", case_dir + "output.raw");
    bool tiled_valid = validate_result(case_dir + "result_tiled.raw", case_dir + "output.raw");
    
    std::cout << "Naive CUDA result is " << (naive_valid ? "valid" : "invalid") << "\n";
    std::cout << "Tiled CUDA result is " << (tiled_valid ? "valid" : "invalid") << "\n";

    // Clean up
    delete[] h_A;
    delete[] h_B;
    delete[] h_C_naive;
    delete[] h_C_tiled;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}