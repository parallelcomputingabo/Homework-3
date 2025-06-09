#include <iostream>
#include <fstream>
#include <string>
#include <cuda_runtime.h>
#include <cmath>
#include <iomanip>
#include <sstream>

#define TILE_WIDTH 32 // Optimized for A100 GPUs
#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Naive CUDA kernel
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

// Tiled CUDA kernel
__global__ void tiled_cuda_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p, uint32_t tile_width) {
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * tile_width + ty;
    int col = blockIdx.x * tile_width + tx;
    
    float sum = 0.0f;
    int num_tiles = (n + tile_width - 1) / tile_width;
    
    for (int t = 0; t < num_tiles; t++) {
        if (row < m && (t * tile_width + tx) < n) {
            As[ty][tx] = A[row * n + t * tile_width + tx];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if (col < p && (t * tile_width + ty) < n) {
            Bs[ty][tx] = B[(t * tile_width + ty) * p + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        for (int k = 0; k < tile_width; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < m && col < p) {
        C[row * p + col] = sum;
    }
}

// Read matrix from text file
float* read_matrix(const std::string& filename, uint32_t& rows, uint32_t& cols) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    
    file >> rows >> cols;
    float* matrix = new float[rows * cols];
    for (uint32_t i = 0; i < rows * cols; i++) {
        file >> matrix[i];
    }
    
    file.close();
    return matrix;
}

// Write matrix to text file
void write_matrix(const std::string& filename, float* matrix, uint32_t rows, uint32_t cols) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    
    file << rows << " " << cols << "\n";
    for (uint32_t i = 0; i < rows; i++) {
        for (uint32_t j = 0; j < cols; j++) {
            file << std::fixed << std::setprecision(2) << matrix[i * cols + j]; // Changed to 2 decimal places
            if (j < cols - 1) file << " ";
        }
        file << "\n";
    }
    
    file.close();
}

// Validate result against reference file
bool validate_result(const std::string& result_file, const std::string& reference_file) {
    std::ifstream res_file(result_file);
    std::ifstream ref_file(reference_file);
    if (!res_file.is_open() || !ref_file.is_open()) {
        std::cerr << "Error opening result or reference file" << std::endl;
        return false;
    }
    
    std::string line;
    uint32_t res_rows, res_cols, ref_rows, ref_cols;
    
    // Read dimensions from result file
    std::getline(res_file, line);
    std::istringstream res_dim(line);
    res_dim >> res_rows >> res_cols;
    
    // Read dimensions from reference file
    std::getline(ref_file, line);
    std::istringstream ref_dim(line);
    ref_dim >> ref_rows >> ref_cols;
    
    if (res_rows != ref_rows || res_cols != ref_cols) {
        std::cerr << "Dimension mismatch: result (" << res_rows << "x" << res_cols 
                  << ") vs reference (" << ref_rows << "x" << ref_cols << ")" << std::endl;
        return false;
    }
    
    float res_val, ref_val;
    float tolerance = 1e-3f; // Increased tolerance to handle floating-point differences
    uint32_t index = 0;
    bool has_error = false;
    
    // Read values line by line to handle varying whitespace
    for (uint32_t i = 0; i < res_rows; i++) {
        std::getline(res_file, line);
        std::istringstream res_stream(line);
        std::getline(ref_file, line);
        std::istringstream ref_stream(line);
        
        for (uint32_t j = 0; j < res_cols; j++, index++) {
            if (!(res_stream >> res_val) || !(ref_stream >> ref_val)) {
                std::cerr << "Error reading values at index " << index 
                          << " (row " << i << ", col " << j << ")" << std::endl;
                return false;
            }
            
            if (std::abs(res_val - ref_val) > tolerance) {
                std::cerr << "Validation failed at index " << index 
                          << " (row " << i << ", col " << j << "): " 
                          << std::fixed << std::setprecision(2) << res_val << " vs " 
                          << std::fixed << std::setprecision(2) << ref_val 
                          << " (diff: " << std::abs(res_val - ref_val) << ")" << std::endl;
                has_error = true;
            }
        }
    }
    
    // Check for extra data
    if (std::getline(res_file, line) || std::getline(ref_file, line)) {
        std::cerr << "Extra data found in result or reference file" << std::endl;
        return false;
    }
    
    res_file.close();
    ref_file.close();
    
    if (has_error) {
        return false;
    }
    
    return true;
}

int main(int argc, char* argv[]) {
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
    
    // Read matrices
    uint32_t m, n, n2, p;
    float* h_A = read_matrix(input0_file, m, n);
    float* h_B = read_matrix(input1_file, n2, p);
    if (n != n2) {
        std::cerr << "Matrix dimensions incompatible: A(" << m << "x" << n << ") B(" << n2 << "x" << p << ")" << std::endl;
        delete[] h_A;
        delete[] h_B;
        return 1;
    }
    
    float* h_C = new float[m * p];
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA_ERROR(cudaMalloc(&d_A, m * n * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_B, n * p * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_C, m * p * sizeof(float)));
    
    // CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    
    // Grid and block dimensions
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((p + TILE_WIDTH - 1) / TILE_WIDTH, (m + TILE_WIDTH - 1) / TILE_WIDTH);
    
    // Naive CUDA
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, m * n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, n * p * sizeof(float), cudaMemcpyHostToDevice));
    naive_cuda_matmul<<<gridDim, blockDim>>>(d_C, d_A, d_B, m, n, p);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaMemcpy(h_C, d_C, m * p * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    
    float naive_time;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&naive_time, start, stop));
    naive_time /= 1000.0f; // Convert to seconds
    
    // Write and validate naive result
    write_matrix(result_file, h_C, m, p);
    bool naive_correct = validate_result(result_file, reference_file);
    
    // Tiled CUDA
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, m * n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, n * p * sizeof(float), cudaMemcpyHostToDevice));
    tiled_cuda_matmul<<<gridDim, blockDim>>>(d_C, d_A, d_B, m, n, p, TILE_WIDTH);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaMemcpy(h_C, d_C, m * p * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    
    float tiled_time;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&tiled_time, start, stop));
    tiled_time /= 1000.0f; // Convert to seconds
    
    // Write and validate tiled result
    write_matrix(result_file, h_C, m, p);
    bool tiled_correct = validate_result(result_file, reference_file);
    
    // Print results
    std::cout << "Case " << case_number << " (" << m << "x" << n << "x" << p << "):\n";
    std::cout << "Naive CUDA time: " << naive_time << " seconds\n";
    std::cout << "Tiled CUDA time: " << tiled_time << " seconds\n";
    std::cout << "Naive result: " << (naive_correct ? "PASS" : "FAIL") << "\n";
    std::cout << "Tiled result: " << (tiled_correct ? "PASS" : "FAIL") << "\n";
    std::cout << "Speedup (Tiled vs Naive): " << (naive_time / tiled_time) << "x\n";
    
    // Output for performance table
    std::ofstream perf_file("performance.txt", std::ios::app);
    perf_file << case_number << "\t" << m << "x" << n << "x" << p << "\t"
              << naive_time << "\t" << tiled_time << "\t" << (naive_time / tiled_time) << "\n";
    perf_file.close();
    
    // Cleanup
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    
    return 0;
}