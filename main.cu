#include <iostream>
#include <fstream>
#include <format>
#include <string>
#include <cmath>
#include <filesystem>
#include <cuda_runtime.h>

using namespace std;
namespace fs = filesystem;

__global__ void naive_cuda_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p)
{
    // Implement naive CUDA matrix multiplication
    // Each thread computes one element of the output matrix C
    uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < p)
    {
        float value = 0.0f;
        // Compute the dot product for the element C[row][col]
        for (uint32_t k = 0; k < n; k++)
        {
            value += A[row * n + k] * B[k * p + col];
        }
        // Store the result in C
        C[row * p + col] = value;
    }
}

__global__ void tiled_cuda_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p, uint32_t tile_width)
{
    // Implement tiled CUDA matrix multiplication

    // Each block computes a tile of the output matrix C
    uint32_t row = blockIdx.y * tile_width + threadIdx.y;
    uint32_t col = blockIdx.x * tile_width + threadIdx.x;

    // Shared memory for tiles of A and B
    extern __shared__ float shared_mem[];
    float *A_shared = shared_mem;
    float *B_shared = (float *)&A_shared[tile_width * tile_width];

    float value = 0.0f;

    // Loop over tiles
    // Each tile is of size tile_width x tile_width
    for (uint32_t tile = 0; tile < (n + tile_width - 1) / tile_width; tile++)
    {
        // Load tiles of A from global memory into shared memory
        if (row < m && tile * tile_width + threadIdx.x < n)
        {
            A_shared[threadIdx.y * tile_width + threadIdx.x] = A[row * n + tile * tile_width + threadIdx.x];
        }
        else
        {
            // If the row is out of bounds, fill with zero
            A_shared[threadIdx.y * tile_width + threadIdx.x] = 0.0f;
        }

        // Load tiles of B from global memory into shared memory
        if (col < p && tile * tile_width + threadIdx.y < n)
        {
            B_shared[threadIdx.y * tile_width + threadIdx.x] = B[(tile * tile_width + threadIdx.y) * p + col];
        }
        else
        {
            // If the column is out of bounds, fill with zero
            B_shared[threadIdx.y * tile_width + threadIdx.x] = 0.0f;
        }

        // Synchronize threads to ensure all threads have finished copying before calculation
        __syncthreads();

        // Compute the dot product for the element C[row][col]
        for (uint32_t k = 0; k < tile_width; k++)
        {
            value += A_shared[threadIdx.y * tile_width + k] * B_shared[k * tile_width + threadIdx.x];
        }

        // Synchronize threads to ensure all threads have finished calculating before loading the next tile
        __syncthreads();
    }
    if (row < m && col < p)
    {
        // Store the result in C
        C[row * p + col] = value;
    }
}

int read_matrix_dimensions(fs::path path, uint32_t &rows, uint32_t &cols)
{
    ifstream file(path);
    if (file.is_open())
    {
        file >> rows >> cols;
        file.close();
    }
    else
    {
        return -1;
    }

    return 0;
}

int read_matrix_elements(fs::path path, float *matrix, uint32_t rows, uint32_t cols)
{
    ifstream file(path);
    if (file.is_open())
    {

        int dummy1, dummy2;
        file >> dummy1 >> dummy2; // Skip the dimensions

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                file >> matrix[i * cols + j];
            }
        }

        file.close();
    }
    else
    {
        return -1;
    }

    return 0;
}

int write_matrix_result(fs::path filePath, const float *matrix, uint32_t rows, uint32_t cols)
{
    ofstream file(filePath);
    if (file.is_open())
    {
        // Write rows and columns
        file << rows << " " << cols << "\n";

        for (uint32_t i = 0; i < rows; i++)
        {
            for (uint32_t j = 0; j < cols; j++)
            {
                file << matrix[i * cols + j] << " ";
            }
            file << "\n"; // Newline after each row
        }

        file.close();
    }
    else
    {
        return -1;
    }

    return 0;
}

bool validate_result(fs::path result_file, fs::path reference_file)
{
    int result;
    uint32_t m, p;

    result = read_matrix_dimensions(result_file, m, p);
    if (result == -1)
    {
        cerr << "Failed to open file: " << result_file << endl;
        return 1;
    }

    float *C_result = new float[m * p];
    float *C_reference = new float[m * p];

    result = read_matrix_elements(result_file, C_result, m, p);
    if (result == -1)
    {
        cerr << "Failed to open file: " << result_file << endl;
        exit(1);
    }
    result = read_matrix_elements(result_file, C_reference, m, p);
    if (result == -1)
    {
        cerr << "Failed to open file: " << reference_file << endl;
        exit(1);
    }

    bool same = true;
    float c_res, c_ref;

    for (uint32_t i = 0; i < m; i++)
    {
        for (uint32_t j = 0; j < p; j++)
        {
            // Using a comparison with rounding because of precision error for floating point numbers
            c_res = C_result[i * p + j];
            c_ref = C_reference[i * p + j];
            c_res = std::round(c_res * 100.0f) / 100.0f;
            c_ref = std::round(c_ref * 100.0f) / 100.0f;
            if (c_res != c_ref)
            {
                cout << format("Difference at row = {} col = {} element1 = {} element2 = {}", i, j, c_res, c_ref) << endl;
                same = false;
                break;
            }
        }

        if (same == false)
            break;
    }

    delete[] C_result;
    delete[] C_reference;

    return same;
}

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        cerr << "Usage: " << argv[0] << " <case_number>" << endl;
        return -1;
    }

    int case_number = atoi(argv[1]);
    if (case_number < 0 || case_number > 9)
    {
        cerr << "Case number must be between 0 and 9" << endl;
        return -1;
    }

    // Construct file paths
    fs::path folder = fs::path(SOURCE_DIR) / "data" / to_string(case_number);
    fs::path input0_file = folder / "input0.raw";
    fs::path input1_file = folder / "input1.raw";
    fs::path result_file = folder / "result.raw";
    fs::path reference_file = folder / "output.raw";

    uint32_t m, n, p;
    // A is m x n, B is n x p, C is m x p

    // Read input0.raw (matrix A)
    int result = read_matrix_dimensions(input0_file, m, n);
    if (result == -1)
    {
        cerr << "Failed to open file: " << input0_file << endl;
        return -1;
    }

    float *A = new float[m * n];

    result = read_matrix_elements(input0_file, A, m, n);
    if (result == -1)
    {
        cerr << "Failed to open file: " << input0_file << endl;
        return -1;
    }

    // Read input1.raw (matrix B)
    result = read_matrix_dimensions(input1_file, n, p);
    if (result == -1)
    {
        cerr << "Failed to open file: " << input1_file << endl;
        return -1;
    }

    float *B = new float[n * p];

    result = read_matrix_elements(input1_file, B, n, p);
    if (result == -1)
    {
        cerr << "Failed to open file: " << input1_file << endl;
        return -1;
    }

    // Allocate memory for matrices
    float *C_host = new float[m * p];
    float *C_naive_dev = nullptr;
    float *C_tiled_dev = nullptr;
    float *A_dev = nullptr;
    float *B_dev = nullptr;

    // Use cudaMalloc and cudaMemcpy for GPU memory
    cudaError_t err;
    err = cudaMalloc((void **)&C_naive_dev, m * p * sizeof(float));
    if (err != cudaSuccess)
    {
        cerr << "Failed to allocate device memory for C_naive: " << cudaGetErrorString(err) << endl;
        return -1;
    }

    err = cudaMalloc((void **)&C_tiled_dev, m * p * sizeof(float));
    if (err != cudaSuccess)
    {
        cerr << "Failed to allocate device memory for C_tiled: " << cudaGetErrorString(err) << endl;
        return -1;
    }

    err = cudaMalloc((void **)&A_dev, m * n * sizeof(float));
    if (err != cudaSuccess)
    {
        cerr << "Failed to allocate device memory for A: " << cudaGetErrorString(err) << endl;
        return -1;
    }

    err = cudaMalloc((void **)&B_dev, n * p * sizeof(float));
    if (err != cudaSuccess)
    {
        cerr << "Failed to allocate device memory for B: " << cudaGetErrorString(err) << endl;
        return -1;
    }

    // Copy matrices A and B to device
    err = cudaMemcpy(A_dev, A, m * n * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        cerr << "Failed to copy A to device: " << cudaGetErrorString(err) << endl;
        return -1;
    }

    err = cudaMemcpy(B_dev, B, n * p * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        cerr << "Failed to copy B to device: " << cudaGetErrorString(err) << endl;
        return -1;
    }

    // Measure naive CUDA performance
    float naive_cuda_time = 0.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Launch naive_cuda_matmul kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((p + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y);

    naive_cuda_matmul<<<gridSize, blockSize>>>(C_naive_dev, A_dev, B_dev, m, n, p);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        cerr << "Failed to launch naive_cuda_matmul kernel: " << cudaGetErrorString(err) << endl;
        return -1;
    }

    // Copy result from device to host
    err = cudaMemcpy(C_host, C_naive_dev, m * p * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        cerr << "Failed to copy C from device to host: " << cudaGetErrorString(err) << endl;
        return -1;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&naive_cuda_time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Write naive CUDA result to file and validate
    result = write_matrix_result(result_file, C_host, m, p);
    if (result == -1)
    {
        cerr << "Failed to write result to file: " << result_file << endl;
        return -1;
    }

    bool is_valid = validate_result(result_file, reference_file);
    if (!is_valid)
    {
        cerr << "Validation failed for naive CUDA result." << endl;
    }

    // Measure tiled CUDA performance
    float tiled_cuda_time = 0.0f;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Launch tiled_cuda_matmul kernel
    uint32_t tile_width = 32;
    dim3 tiled_blockSize(tile_width, tile_width);
    dim3 tiled_gridSize((p + tile_width - 1) / tile_width, (m + tile_width - 1) / tile_width);

    // Shared memory size because of dynamic allocation
    size_t shared_memory_size = tile_width * tile_width * sizeof(float) * 2; // For A and B matrices

    tiled_cuda_matmul<<<tiled_gridSize, tiled_blockSize, shared_memory_size>>>(C_tiled_dev, A_dev, B_dev, m, n, p, tile_width);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        cerr << "Failed to launch tiled_cuda_matmul kernel: " << cudaGetErrorString(err) << endl;
        return -1;
    }

    // Copy result from device to host
    err = cudaMemcpy(C_host, C_tiled_dev, m * p * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        cerr << "Failed to copy C from device to host: " << cudaGetErrorString(err) << endl;
        return -1;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&tiled_cuda_time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Write tiled CUDA result to file and validate
    result = write_matrix_result(result_file, C_host, m, p);
    if (result == -1)
    {
        cerr << "Failed to write result to file: " << result_file << endl;
        return -1;
    }

    is_valid = validate_result(result_file, reference_file);
    if (!is_valid)
    {
        cerr << "Validation failed for tiled CUDA result." << endl;
    }

    // Print performance results
    std::cout << "Case " << case_number << " (" << m << "x" << n << "x" << p << "):\n";
    std::cout << "Naive CUDA time: " << naive_cuda_time << " seconds\n";
    std::cout << "Tiled CUDA time: " << tiled_cuda_time << " seconds\n";
    std::cout << "Speedup: " << (naive_cuda_time / tiled_cuda_time) << "x\n";

    // Clean up
    delete[] A;
    delete[] B;
    delete[] C_host;
    cudaFree(C_naive_dev);
    cudaFree(C_tiled_dev);
    cudaFree(A_dev);
    cudaFree(B_dev);
    cudaDeviceReset();
    if (cudaGetLastError() != cudaSuccess)
    {
        cerr << "CUDA error: " << cudaGetErrorString(cudaGetLastError()) << endl;
        return -1;
    }

    return 0;
}