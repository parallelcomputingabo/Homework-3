#include <iostream>
#include <fstream>
#include <string>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include <filesystem>

using namespace std;
namespace fs = std::filesystem;


__global__ void naive_cuda_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p) {
    // Compute the global row and column indices for this thread
    uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;

    // Only process elements within the matrix boundaries
    if (row < m && col < p) {
        float product = 0.0f;

        // Accumulate the result of the dot product for C[row][col]
        for (uint32_t k = 0; k < n; ++k) {
            product += A[row * n + k] * B[k * p + col];
        }

        // Store the final result in matrix C
        C[row * p + col] = product;
    }
}


__global__ void tiled_cuda_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p, uint32_t tile_width) {
    // Determine the row and column this thread computes in the output matrix
    uint32_t row = blockIdx.y * tile_width + threadIdx.y;
    uint32_t col = blockIdx.x * tile_width + threadIdx.x;

    // Declare shared memory: one buffer for A tiles, one for B tiles
    extern __shared__ float shared_mem[];
    float *tile_A = shared_mem;
    float *tile_B = &tile_A[tile_width * tile_width];

    float sum = 0.0f;  // Temporary accumulator for the output element

    // Iterate over all tiles needed to compute C[row][col]
    for (uint32_t tile = 0; tile < (n + tile_width - 1) / tile_width; ++tile) {
        // Load a tile of A from global to shared memory, with bounds check
        if (row < m && tile * tile_width + threadIdx.x < n) {
            tile_A[threadIdx.y * tile_width + threadIdx.x] = A[row * n + tile * tile_width + threadIdx.x];
        } else {
            tile_A[threadIdx.y * tile_width + threadIdx.x] = 0.0f;
        }

        // Load a tile of B from global to shared memory, with bounds check
        if (col < p && tile * tile_width + threadIdx.y < n) {
            tile_B[threadIdx.y * tile_width + threadIdx.x] = B[(tile * tile_width + threadIdx.y) * p + col];
        } else {
            tile_B[threadIdx.y * tile_width + threadIdx.x] = 0.0f;
        }

        // Wait until all threads have finished loading data
        __syncthreads();

        // Multiply the loaded tiles and accumulate the result
        for (uint32_t k = 0; k < tile_width; ++k) {
            sum += tile_A[threadIdx.y * tile_width + k] * tile_B[k * tile_width + threadIdx.x];
        }

        // Synchronize before loading the next tile
        __syncthreads();
    }

    // Write the computed value to global memory if inside bounds
    if (row < m && col < p) {
        C[row * p + col] = sum;
    }
}



bool validate_result(float* A, float* B, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            // Round both values to two decimal places
            float a = round(A[i * cols + j] * 100.0f) / 100.0f;
            float b = round(B[i * cols + j] * 100.0f) / 100.0f;

            // Compare the rounded values
            if (a != b) {
                std::cout << "Validation Failed" << std::endl;
                return false;
            }
        }
    }

    std::cout << "Validation Passed" << std::endl;
    return true;
}



int main(int argc, char *argv[]) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <case_number>" << endl;
        return -1;
    }

    int case_number = atoi(argv[1]);
    if (case_number < 0 || case_number > 9) {
        cerr << "Case number must be between 0 and 9" << endl;
        return -1;
    }

    cout << "case_number: " << case_number << endl;
    // Construct file paths
    string folder = "../data/" + to_string(case_number) + "/";
    string input0_file = folder + "input0.raw";
    string input1_file = folder + "input1.raw";
    string result_file = folder + "result.raw";
    string reference_file = folder + "output.raw";

    cout << "Opening A from: " << input0_file << endl;
    cout << "Opening B from: " << input1_file << endl;
    cout << "Writing result to: " << result_file << endl;


    int m, n, p, dummy;

    ifstream fileA(input0_file);
    ifstream fileB(input1_file);

    fileA >> m >> n;       // A is m x n
    fileB >> dummy >> p;   // B is n x p

    cout << "Dimensions of A: " << m << " x " << n << endl;
    cout << "Dimensions of B: " << dummy << " x " << p << endl;

    // TODO Read input0.raw (matrix A)
    float* A = new float[m * n];

    for (int i = 0; i < m * n; ++i) {
        fileA >> A[i];
    }
    fileA.close();


    // TODO Read input1.raw (matrix B)
    float* B = new float[n * p];

    for (int i = 0; i < n * p; ++i) {
        fileB >> B[i];
    }
    fileB.close();
    // Allocate host memory for the result matrix
    float* C_host = new float[m * p];

    // Declare device pointers
    float *A_dev, *B_dev, *C_naive_dev, *C_tiled_dev;

    // Allocate device memory for input and output matrices
    cudaMalloc(&A_dev, m * n * sizeof(float));
    cudaMalloc(&B_dev, n * p * sizeof(float));
    cudaMalloc(&C_naive_dev, m * p * sizeof(float));
    cudaMalloc(&C_tiled_dev, m * p * sizeof(float));

    // Transfer input matrices A and B to device
    cudaMemcpy(A_dev, A, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_dev, B, n * p * sizeof(float), cudaMemcpyHostToDevice);

    // Configure execution parameters for naive kernel
    dim3 block_size_naive(16, 16);

    dim3 grid_size_naive((p + block_size_naive.x - 1) / block_size_naive.x,
                     (m + block_size_naive.y - 1) / block_size_naive.y);

    // Launch and time the naive kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Launch naive matrix multiplication kernel
    naive_cuda_matmul<<<grid_size_naive, block_size_naive>>>(C_naive_dev, A_dev, B_dev, m, n, p);

    // Copy result from device to host
    cudaMemcpy(C_host, C_naive_dev, m * p * sizeof(float), cudaMemcpyDeviceToHost);

    // Finish timing and compute duration
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float naive_time = 0.0f;
    cudaEventElapsedTime(&naive_time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);


    ofstream file_out_naive(result_file);
    if (!file_out_naive.is_open()) {
        cerr << "Failed to open file: " << result_file << endl;
        return -1;
    }

    file_out_naive << m << " " << p << endl;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) {
            file_out_naive << C_host[i * p + j];
            if (j != p - 1) file_out_naive << " ";
        }
        if (i != m - 1) file_out_naive << endl;
    }
    file_out_naive.close();

    // Load reference file (output.raw)
    ifstream ref(reference_file);
    if (!ref.is_open()) {
        cerr << "Failed to open reference file: " << reference_file << endl;
        return -1;
    }

    int ref_m, ref_p;
    ref >> ref_m >> ref_p;

    float* C_reference = new float[ref_m * ref_p];
    for (int i = 0; i < ref_m * ref_p; ++i) {
        ref >> C_reference[i];
    }
    ref.close();

    // Validate
    bool valid_naive = validate_result(C_host, C_reference, m, p);

    delete[] C_reference;



    // ===== Tiled kernel
    // Configure tiled kernel launch parameters
    uint32_t tile_width = 32;
    dim3 block_size_tiled(tile_width, tile_width);
    dim3 grid_size_tiled((p + tile_width - 1) / tile_width,
                     (m + tile_width - 1) / tile_width);

    size_t shared_memory_bytes = 2 * tile_width * tile_width * sizeof(float); // For tiles of A and B

    // Start timing for tiled CUDA execution
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Launch the tiled matrix multiplication kernel
    tiled_cuda_matmul<<<grid_size_tiled, block_size_tiled, shared_memory_bytes>>>(C_tiled_dev, A_dev, B_dev, m, n, p, tile_width);

    // Transfer the result back to host memory
    cudaMemcpy(C_host, C_tiled_dev, m * p * sizeof(float), cudaMemcpyDeviceToHost);

    // Stop timing and compute elapsed time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float tiled_time = 0.0f;
    cudaEventElapsedTime(&tiled_time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);


    ofstream file_out_tiled(result_file);
    if (!file_out_tiled.is_open()) {
        cerr << "Failed to open file: " << result_file << endl;
        return -1;
    }

    file_out_tiled << m << " " << p << endl;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) {
            file_out_tiled << C_host[i * p + j];
            if (j != p - 1) file_out_tiled << " ";
        }
        if (i != m - 1) file_out_tiled << endl;
    }
    file_out_tiled.close();

    // Load reference file again (output.raw)
    ifstream ref2(reference_file);
    if (!ref2.is_open()) {
        cerr << "Failed to open reference file: " << reference_file << endl;
        return -1;
    }

    int ref_m2, ref_p2;
    ref2 >> ref_m2 >> ref_p2;

    float* C_reference2 = new float[ref_m2 * ref_p2];
    for (int i = 0; i < ref_m2 * ref_p2; ++i) {
        ref2 >> C_reference2[i];
    }
    ref2.close();

    // Validate
    bool valid_tiled = validate_result(C_host, C_reference2, m, p);

    delete[] C_reference2;

     // Print performance results
    cout << "Case " << case_number << " (" << m << "x" << n << "x" << p << "):\n";
    cout << "Naive CUDA time: " << naive_time / 1000.0<< " seconds\n";
    cout << "Tiled CUDA time: " << tiled_time / 1000.0<< " seconds\n";

    // Cleanup
    delete[] A;
    delete[] B;
    delete[] C_host;
    cudaFree(A_dev);
    cudaFree(B_dev);
    cudaFree(C_naive_dev);
    cudaFree(C_tiled_dev);
    cudaDeviceReset();

    return 0;
}