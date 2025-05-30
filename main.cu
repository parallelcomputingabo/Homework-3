#include <iostream>
#include <fstream>
#include <string>
#include <cuda_runtime.h>
#include <omp.h>
#include <iomanip>


__global__ void naive_cuda_matmul(float *C, float *A, float *B, int m, int n, int p) {
    // Implement naive CUDA matrix multiplication
    // Define row and col size
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Iterate block
    if (row < m && col < p) {
        float value = 0.0f;
        for (int i = 0; i < n; ++i) {
            // Calculate value of C by multiplying cells from A and B
            value += A[row * n + i] * B[i * p + col];
        }
        // Set calculated value to C
        C[row * p + col] = value;
    }
}

__global__ void tiled_cuda_matmul(float *C, float *A, float *B, int m, int n, int p, int tile_width) {
    // Implement tiled CUDA matrix multiplication
    extern __shared__ float sharedMemory[];
    float *sA = sharedMemory;
    float *sB = &sharedMemory[tile_width * tile_width];

    // Define row and col size
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float value = 0.0f;

    for (int i = 0; i < n; i += tile_width) {
        // Load tiles into shared memory for faster calculations
        if (row < m && (threadIdx.x + i) < n) {
            sA[threadIdx.y * tile_width + threadIdx.x] = A[row * n + (threadIdx.x + i)];
        } else {
            sA[threadIdx.y * tile_width + threadIdx.x] = 0.0f;
        }
        // Same with matrix B
        if (col < p && (threadIdx.y + i) < n) {
            sB[threadIdx.y * tile_width + threadIdx.x] = B[(threadIdx.y + i) * p + col];
        } else {
            sB[threadIdx.y * tile_width + threadIdx.x] = 0.0f;
        }

        // Sync to ensure all threads in a block have loaded data before computation
        __syncthreads();

        for (int i = 0; i < tile_width; ++i) {
            // Calculate value of C by multiplying tiles from shared memory A and B
            value += sA[threadIdx.y * tile_width + i] * sB[i * tile_width + threadIdx.x];
        }

        __syncthreads();
    }
    // Set calculated value to C
    if (row < m && col < p) {
        C[row * p + col] = value;
    }
}

bool validate_result(const std::string &result_file, const std::string &reference_file, int m, int p) {
    // Implement result validation (same as Assignment 2)
    std::ifstream comparison(reference_file);
    if (!comparison.is_open()) {
        // Validate that file opened correctly
        std::cerr << "Unable to open file";
        exit(1);
    }

    std::ifstream res(result_file);
    if (!res.is_open()) {
        // Validate that file opened correctly
        std::cerr << "Unable to open file";
        exit(1);
    }

    float Comp, ResValue;

    // Iterate using the dimensions of C.
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) {
            // Get element from both matrix by index, compare values and throw error if values don't match
            res >> ResValue;
            comparison >> Comp;
            if (ResValue != Comp) {
                std::cerr << "Value mismatch";
                exit(1);
            }
        }
    }

    // Close both files once comparison is done
    comparison.close();
    res.close();
    return true;
}

int main(int argc, char *argv[]) {
    // Read input0.raw (matrix A) and input1.raw (matrix B)
    int m, n, p;
    
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

    // Read input0.raw (matrix A)
    std::ifstream FileA(input0_file);
    // Validate that file is opened correctly
    if (!FileA.is_open()) {
        std::cerr << "Error opening file";
        return 1;
    }

    // Read input1.raw (matrix B)
    std::ifstream FileB(input1_file);
    // Validate that file is opened correctly
    if (!FileB.is_open()) {
        std::cerr << "Error opening file";
        return 1;
    }

    // Get matrix dimensions
    FileA >> m >> n;
    FileB >> n >> p;

    
    // Allocate memory for matrices A and B, to read data from files
    float* A = (float*)malloc(m * n * sizeof(float));
    // Validate that memory is allocated correctly
    if (A == NULL) {
        std::cerr << "Memory allocation failed";
        return 1;
    }

    float* B = (float*)malloc(n * p * sizeof(float));
    if (B == NULL) {
        std::cerr << "Memory allocation failed";
        return 1;
    }

    //Read matrix elements into A and B (row-major order), close file after reading
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            FileA >> A[i * n + j];
        }
    }
    FileA.close();

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < p; ++j) {
            FileB >> B[i * p + j];
        }
    }
    FileB.close();

    // Allocate memory for result matrices
    float *C_naive = new float[m * p];
    float *C_tiled = new float[m * p];

    // Use cudaMalloc and cudaMemcpy for GPU memory
    // Allocate device memory for both naive and tiled
    float* cu_naive_A;
    float* cu_tiled_A;
    float* cu_naive_B;
    float* cu_tiled_B;
    float* cu_naive_C;
    float* cu_tiled_C;

    cudaMalloc((void**)&cu_naive_A, m * n * sizeof(float));
    cudaMalloc((void**)&cu_tiled_A, m * n * sizeof(float));
    cudaMalloc((void**)&cu_naive_B, n * p * sizeof(float));
    cudaMalloc((void**)&cu_tiled_B, n * p * sizeof(float));
    cudaMalloc((void**)&cu_naive_C, m * p * sizeof(float));
    cudaMalloc((void**)&cu_tiled_C, m * p * sizeof(float));

    // Measure naive CUDA performance
    // Create cuda events and start timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Copy data from initial matrices
    cudaMemcpy(cu_naive_A, A, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cu_naive_B, B, n * p * sizeof(float), cudaMemcpyHostToDevice);
    
    // Set tile size,
    int TILE_SIZE = 16;
    // block size and number of blocks.
    dim3 numThreadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((p + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x, (m + numThreadsPerBlock.y - 1) / numThreadsPerBlock.y);

    // Launch naive_cuda_matmul kernel
    naive_cuda_matmul<<< numBlocks, numThreadsPerBlock >>> (cu_naive_C, cu_naive_A, cu_naive_B, m, n, p);
    
    // Copy results back from device to host
    cudaMemcpy(C_naive, cu_naive_C, m * p * sizeof(float), cudaMemcpyDeviceToHost);

    // Stop timing, write down result and destroy events
    cudaEventRecord(stop);
    float naive_cuda_time = 0;
    cudaEventElapsedTime(&naive_cuda_time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Write naive CUDA result to file and validate
    // Write dimensions and elements to result.raw
    std::ofstream result(result_file);
    // Validate that file is created correctly
    if (!result) {
        std::cerr << "Unable to open file";
        exit(1);
    }

    // Write the dimensions of C on the first line
    result << m << " " << p << std::endl;
    result << std::fixed << std::setprecision(2);
    // Iterate C and write each element to result.raw
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) {
            result << C_naive[i * p + j] << " ";
        }
        result << std::endl;
    }
    // Close file after writing
    result.close();

    // Validate naive CUDA results
    bool naive_correct = validate_result(result_file, reference_file, m, p);
    if (!naive_correct) {
        std::cerr << "Naive result validation failed for case " << case_number << std::endl;
    }

    // Measure tiled CUDA performance
    // Create cuda events and start timing
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Copy data from initial matrices
    cudaMemcpy(cu_tiled_A, A, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cu_tiled_B, B, n * p * sizeof(float), cudaMemcpyHostToDevice);
    size_t sharedMemorySize = 2 * TILE_SIZE * TILE_SIZE * sizeof(float);

    // Launch tiled_cuda_matmul kernel
    tiled_cuda_matmul<<< numBlocks, numThreadsPerBlock, sharedMemorySize >>> (cu_tiled_C, cu_tiled_A, cu_tiled_B, m, n, p, TILE_SIZE);
    
    // Copy results back from device to host
    cudaMemcpy(C_tiled, cu_tiled_C, m * p * sizeof(float), cudaMemcpyDeviceToHost);
    // Stop timing, write down results and destroy events
    cudaEventRecord(stop);
    float tiled_cuda_time = 0;
    cudaEventElapsedTime(&tiled_cuda_time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Free memory of initial matrices, they are no longer needed at this point
    free(A);
    free(B);

    // Write tiled CUDA result to file and validate
    std::ofstream result_tiled(result_file);
    // Validate that file is created correctly
    if (!result_tiled) {
        std::cerr << "Unable to open file";
        exit(1);
    }

    // Write the dimensions of C on the first line
    result_tiled << m << " " << p << std::endl;
    result_tiled << std::fixed << std::setprecision(2);
    // Iterate C and write each element to result.raw
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) {
            result_tiled << C_tiled[i * p + j] << " ";
        }
        result_tiled << std::endl;
    }
    // Close file after writing
    result_tiled.close();

    // Validate tiled CUDA results
    bool tiled_correct = validate_result(result_file, reference_file, m, p);
    if (!tiled_correct) {
        std::cerr << "Tiled result validation failed for case " << case_number << std::endl;
    }

    // Print performance results
    std::cout << "Case " << case_number << " (" << m << "x" << n << "x" << p << "):\n";
    std::cout << "Naive CUDA time: " << naive_cuda_time << " seconds\n";
    std::cout << "Tiled CUDA time: " << tiled_cuda_time << " seconds\n";

    // Clean up
    cudaFree(cu_naive_A);
    cudaFree(cu_tiled_A);
    cudaFree(cu_naive_B);
    cudaFree(cu_tiled_B);
    cudaFree(cu_naive_C);
    cudaFree(cu_tiled_C);
    delete[] C_naive;
    delete[] C_tiled;
    return 0;
}