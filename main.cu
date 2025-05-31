#include <iostream>
#include <fstream>
#include <string>
#include <cuda_runtime.h>
#include <cstdint>

#define TILE_WIDTH 16
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

__global__ void naive_cuda_matmul(double *C, double *A, double *B, uint32_t m, uint32_t n, uint32_t p) {
    // Calculate thread indices
    uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check bounds
    if (row < m && col < p) {
        double sum = 0.0f;
        
        // Compute dot product for C[row][col]
        for (uint32_t k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * p + col];
        }
        
        C[row * p + col] = sum;
    }
}

__global__ void tiled_cuda_matmul(double *C, double *A, double *B, uint32_t m, uint32_t n, uint32_t p, uint32_t tile_width) {
    // Shared memory for tiles
    extern __shared__ double shared_mem[];
    double* tile_A = shared_mem;
    double* tile_B = &shared_mem[tile_width * tile_width];
    
    // Thread indices
    uint32_t tx = threadIdx.x;
    uint32_t ty = threadIdx.y;
    uint32_t row = blockIdx.y * tile_width + ty;
    uint32_t col = blockIdx.x * tile_width + tx;
    
    double sum = 0.0f;
    
    // Loop over tiles
    for (uint32_t tile = 0; tile < (n + tile_width - 1) / tile_width; ++tile) {
        // Load tile of A into shared memory
        uint32_t a_col = tile * tile_width + tx;
        if (row < m && a_col < n) {
            tile_A[ty * tile_width + tx] = A[row * n + a_col];
        } else {
            tile_A[ty * tile_width + tx] = 0.0f;
        }
        
        // Load tile of B into shared memory
        uint32_t b_row = tile * tile_width + ty;
        if (b_row < n && col < p) {
            tile_B[ty * tile_width + tx] = B[b_row * p + col];
        } else {
            tile_B[ty * tile_width + tx] = 0.0f;
        }
        
        // Synchronize to ensure all data is loaded
        __syncthreads();
        
        // Compute partial dot product
        for (uint32_t k = 0; k < tile_width; ++k) {
            sum += tile_A[ty * tile_width + k] * tile_B[k * tile_width + tx];
        }
        
        // Synchronize before loading next tile
        __syncthreads();
    }
    
    // Write result
    if (row < m && col < p) {
        C[row * p + col] = sum;
    }
}

// Read a matrix from text file (row-major)
double* read_matrix(const std::string& path, uint32_t& rows, uint32_t& cols) {
    std::ifstream in(path);
    if (!in) {
        std::cerr << "Error: cannot open file " << path << std::endl;
        std::exit(EXIT_FAILURE);
    }
    in >> rows >> cols;
    double* mat = new double[static_cast<size_t>(rows) * cols];
    for (uint32_t i = 0; i < rows * cols; ++i) {
        double temp;
        in >> temp;
        mat[i] = static_cast<double>(temp);
    }
    in.close();
    return mat;
}

// Write a matrix to text file (row-major)
void write_matrix(const std::string& path, const double* mat, uint32_t rows, uint32_t cols) {
    std::ofstream out(path);
    if (!out) {
        std::cerr << "Error: cannot write to file " << path << std::endl;
        std::exit(EXIT_FAILURE);
    }
    out << rows << " " << cols << '\n';
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < cols; ++j) {
            out << mat[i * cols + j];
            if (j + 1 < cols) out << ' ';
        }
        out << '\n';
    }
    out.close();
}

// Result validation
bool validate_result(const std::string &result_file, const std::string &reference_file) {
    uint32_t rows1, cols1, rows2, cols2;
    
    double* result = read_matrix(result_file, rows1, cols1);
    double* reference = read_matrix(reference_file, rows2, cols2);
    
    if (rows1 != rows2 || cols1 != cols2) {
        std::cerr << "Matrix dimensions don't match: (" << rows1 << "x" << cols1 
                  << ") vs (" << rows2 << "x" << cols2 << ")" << std::endl;
        delete[] result;
        delete[] reference;
        return false;
    }
    
    const double epsilon = 1e-5f;
    for (uint32_t i = 0; i < rows1 * cols1; ++i) {
        if (std::abs(result[i] - reference[i]) > epsilon) {
            std::cout << "Mismatch at index " << i << ": " 
                      << "Result = " << result[i] << ", Expected = " << reference[i] 
                      << ", Diff = " << std::abs(result[i] - reference[i]) << std::endl;
            delete[] result;
            delete[] reference;
            return false;
        }
    }
    
    delete[] result;
    delete[] reference;
    return true;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <case_number>" << std::endl;
        return EXIT_FAILURE;
    }

    int case_number = std::atoi(argv[1]);
    if (case_number < 0 || case_number > 9) {
        std::cerr << "Case number must be between 0 and 9" << std::endl;
        return EXIT_FAILURE;
    }

    // Construct file paths
    std::string folder = "../data/" + std::to_string(case_number) + "/";
    std::string input0_file = folder + "input0.raw";
    std::string input1_file = folder + "input1.raw";
    std::string result_naive_file = folder + "result_naive_cuda.raw";
    std::string result_tiled_file = folder + "result_tiled_cuda.raw";
    std::string reference_file = folder + "output.raw";

    uint32_t m, n_A, n_B, n, p;
    
    // Read input matrices
    std::cout << "Reading matrix A from: " << input0_file << std::endl;
    double* A_host = read_matrix(input0_file, m, n_A);
    
    std::cout << "Reading matrix B from: " << input1_file << std::endl;
    double* B_host = read_matrix(input1_file, n_B, p);
    
    if (n_A != n_B) {
        std::cerr << "Error: Matrix dimensions do not match for multiplication." << std::endl;
        delete[] A_host;
        delete[] B_host;
        return EXIT_FAILURE;
    }
    n = n_A;
    
    // Allocate host memory for results
    double* C_naive_host = new double[m * p];
    double* C_tiled_host = new double[m * p];
    
    // Allocate GPU memory
    double *A_device, *B_device, *C_device;
    size_t size_A = m * n * sizeof(double);
    size_t size_B = n * p * sizeof(double);  
    size_t size_C = m * p * sizeof(double);
    
    CHECK_CUDA(cudaMalloc(&A_device, size_A));
    CHECK_CUDA(cudaMalloc(&B_device, size_B));
    CHECK_CUDA(cudaMalloc(&C_device, size_C));
    
    // Copy data to GPU
    CHECK_CUDA(cudaMemcpy(A_device, A_host, size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(B_device, B_host, size_B, cudaMemcpyHostToDevice));
    
    // Configure grid and block dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((p + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    // Measure naive CUDA performance
    CHECK_CUDA(cudaEventRecord(start));
    naive_cuda_matmul<<<gridSize, blockSize>>>(C_device, A_device, B_device, m, n, p);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float naive_cuda_time;
    CHECK_CUDA(cudaEventElapsedTime(&naive_cuda_time, start, stop));
    naive_cuda_time /= 1000.0f; // Convert to seconds
    
    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(C_naive_host, C_device, size_C, cudaMemcpyDeviceToHost));
    
    // Write naive CUDA result to file
    std::cout << "Writing naive CUDA result to: " << result_naive_file << std::endl;
    write_matrix(result_naive_file, C_naive_host, m, p);
    
    // Validate naive result
    bool naive_correct = validate_result(result_naive_file, reference_file);
    if (!naive_correct) {
        std::cerr << "Naive CUDA result validation failed for case " << case_number << std::endl;
    }
    
    // Measure tiled CUDA performance
    uint32_t tile_width = TILE_WIDTH;
    dim3 tiled_blockSize(tile_width, tile_width);
    dim3 tiled_gridSize((p + tile_width - 1) / tile_width, (m + tile_width - 1) / tile_width);
    
    // Calculate shared memory size (2 tiles)
    size_t shared_mem_size = 2 * tile_width * tile_width * sizeof(double);
    
    CHECK_CUDA(cudaEventRecord(start));
    tiled_cuda_matmul<<<tiled_gridSize, tiled_blockSize, shared_mem_size>>>(C_device, A_device, B_device, m, n, p, tile_width);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float tiled_cuda_time;
    CHECK_CUDA(cudaEventElapsedTime(&tiled_cuda_time, start, stop));
    tiled_cuda_time /= 1000.0f; // Convert to seconds
    
    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(C_tiled_host, C_device, size_C, cudaMemcpyDeviceToHost));
    
    // Write tiled CUDA result to file
    std::cout << "Writing tiled CUDA result to: " << result_tiled_file << std::endl;
    write_matrix(result_tiled_file, C_tiled_host, m, p);
    
    // Validate tiled result
    bool tiled_correct = validate_result(result_tiled_file, reference_file);
    if (!tiled_correct) {
        std::cerr << "Tiled CUDA result validation failed for case " << case_number << std::endl;
    }
    
    // Print performance results
    std::cout << "Case " << case_number << " (" << m << "x" << n << "x" << p << "):\n";
    std::cout << "Naive CUDA time: " << naive_cuda_time << " seconds\n";
    std::cout << "Tiled CUDA time: " << tiled_cuda_time << " seconds\n";

    // Clean up
    delete[] A_host;
    delete[] B_host;
    delete[] C_naive_host;
    delete[] C_tiled_host;
    
    CHECK_CUDA(cudaFree(A_device));
    CHECK_CUDA(cudaFree(B_device));
    CHECK_CUDA(cudaFree(C_device));
    
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    
    return EXIT_SUCCESS;
}