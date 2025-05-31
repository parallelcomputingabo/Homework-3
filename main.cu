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

__global__ void naive_cuda_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p) {
    // Calculate thread indices
    uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check bounds
    if (row < m && col < p) {
        float sum = 0.0f;
        
        // Compute dot product for C[row][col]
        for (uint32_t k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * p + col];
        }
        
        C[row * p + col] = sum;
    }
}

__global__ void tiled_cuda_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p, uint32_t tile_width) {
    // Shared memory for tiles
    extern __shared__ float shared_mem[];
    float* tile_A = shared_mem;
    float* tile_B = &shared_mem[tile_width * tile_width];
    
    // Thread indices
    uint32_t tx = threadIdx.x;
    uint32_t ty = threadIdx.y;
    uint32_t row = blockIdx.y * tile_width + ty;
    uint32_t col = blockIdx.x * tile_width + tx;
    
    float sum = 0.0f;
    
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
float* read_matrix(const std::string& path, uint32_t& rows, uint32_t& cols) {
    std::ifstream in(path);
    if (!in) {
        std::cerr << "Error: cannot open file " << path << std::endl;
        std::exit(EXIT_FAILURE);
    }
    in >> rows >> cols;
    float* mat = new float[static_cast<size_t>(rows) * cols];
    for (uint32_t i = 0; i < rows * cols; ++i) {
        double temp;
        in >> temp;
        mat[i] = static_cast<float>(temp);
    }
    in.close();
    return mat;
}

// Write a matrix to text file (row-major)
void write_matrix(const std::string& path, const float* mat, uint32_t rows, uint32_t cols) {
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
    
    float* result = read_matrix(result_file, rows1, cols1);
    float* reference = read_matrix(reference_file, rows2, cols2);
    
    if (rows1 != rows2 || cols1 != cols2) {
        std::cerr << "Matrix dimensions don't match: (" << rows1 << "x" << cols1 
                  << ") vs (" << rows2 << "x" << cols2 << ")" << std::endl;
        delete[] result;
        delete[] reference;
        return false;
    }
    
    const float epsilon = 1e-5f;
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

    // TODO: Read input0.raw (matrix A) and input1.raw (matrix B)

    // TODO: Use cudaMalloc and cudaMemcpy for GPU memory

    // Measure naive CUDA performance
    // TODO: Launch naive_cuda_matmul kernel

    // TODO: Write naive CUDA result to file and validate
    // Measure tiled CUDA performance

    // TODO: Launch tiled_cuda_matmul kernel

    // TODO: Write tiled CUDA result to file and validate

    // Print performance results
    std::cout << "Case " << case_number << " (" << m << "x" << n << "x" << p << "):\n";
    std::cout << "Naive CUDA time: " << naive_cuda_time << " seconds\n";
    std::cout << "Tiled CUDA time: " << tiled_cuda_time << " seconds\n";

    // Clean up

    return 0;
}