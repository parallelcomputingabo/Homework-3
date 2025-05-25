#include <iostream>
#include <fstream>
#include <string>
#include <cuda_runtime.h>
#include <sstream>
#include <math_functions.h>

#define TILE_WIDTH 16

__global__ void naive_cuda_matmul(float* C, float* A, float* B, uint32_t m, uint32_t n, uint32_t p) {
    // Calculate the global row (i) and column (j) for this thread
    uint32_t i = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < m && j < p) {
        float temp_C = 0.0f;

        // Sum the dot product
        for (uint32_t k = 0; k < n; ++k) {
            temp_C += A[i * n + k] * B[k * p + j];
        }

        // Round the value to two decimals if the output has more than two decimals
        float temp_100 = temp_C * 100.0f;
        if (floorf(temp_100) != temp_100) {
            C[i * p + j] = roundf(temp_100) / 100.0f;
        }
        else {
            C[i * p + j] = temp_C;
        }
    }
}

__global__ void tiled_cuda_matmul(float* C, float* A, float* B, uint32_t m, uint32_t n, uint32_t p, uint32_t tile_width) {
    __shared__ float A_s[TILE_WIDTH][TILE_WIDTH];
    __shared__ float B_s[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE_WIDTH + ty;
    int col = blockIdx.x * TILE_WIDTH + tx;

    float C_value = 0.0f;

    for (int ph = 0; ph < (n + TILE_WIDTH - 1) / TILE_WIDTH; ++ph) {
        int a_load_row = row;
        int a_load_col = ph * TILE_WIDTH + tx;

        if (a_load_row < m && a_load_col < n) {
            A_s[ty][tx] = A[a_load_row * n + a_load_col];
        }
        else {
            A_s[ty][tx] = 0.0f;
        }

        int b_load_row = ph * TILE_WIDTH + ty;
        int b_load_col = col;

        if (b_load_row < n && b_load_col < p) {
            B_s[ty][tx] = B[b_load_row * p + b_load_col];
        }
        else {
            B_s[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            C_value += A_s[ty][k] * B_s[k][tx];
        }

        __syncthreads();
    }

    if (row < m && col < p) {
        float temp_100 = C_value * 100.0f;
        if (floorf(temp_100) != temp_100) {
            C[row * p + col] = roundf(temp_100) / 100.0f;
        }
        else {
            C[row * p + col] = C_value;
        }
    }
}

/**
 * @brief Writes the result matrix C to a file.
 *
 * Writes matrix dimensions and elements to the specified file.
 *
 * @param path File path for writing the result.
 * @param C Pointer to the result matrix.
 * @param m Number of rows in C.
 * @param p Number of columns in C.
 */
void write_result(const std::string& path, const float* C, int m, int p)
{
    std::ofstream resultFile(path);
    // Handle error
    if (!resultFile)
        return;

    // Write dimensions
    resultFile << m << " " << p << "\n";
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) {
            // Write element
            resultFile << C[i * p + j];
            if (j < p - 1)
                resultFile << " ";
        }
        resultFile << "\n";
    }
    resultFile.close();
}

/**
 * @brief Compares two matrix result files for exact match.
 *
 * Successful test will print "Files match exactly!".
 * Unsuccessful test will print the first mismatch found.
 *
 * @param result_file Path to the generated result matrix file.
 * @param reference_file Path to the expected output matrix file.
 */
bool validate_result(const std::string& result_file, const std::string& reference_file)
{
    std::ifstream result(result_file);
    std::ifstream reference(reference_file);

    // Check if files opened successfully
    if (!result.is_open() || !reference.is_open()) {
        std::cerr << "Validation: Error Opening File" << std::endl;
        return false;
    }

    int m_result = 0, p_result = 0;
    int m_output = 0, p_output = 0;
    std::string line;

    // Read dimensions from result.raw
    if (std::getline(result, line)) {
        std::stringstream ss(line);
        ss >> m_result >> p_result;
    }

    // Read dimensions from output.raw
    if (std::getline(reference, line)) {
        std::stringstream ss(line);
        ss >> m_output >> p_output;
    }

    // Check if dimensions match
    if (m_result != m_output || p_result != p_output) {
        std::cerr << "Validation: Matrix Dimensions Do Not Match!" << std::endl;
        return false;
    }

    // Compare matrices element by element
    float value_result = 0.0f, value_output = 0.0f;
    for (int i = 0; i < m_result; ++i) {
        for (int j = 0; j < p_result; ++j) {
            result >> value_result;
            reference >> value_output;
            if (value_result != value_output) {
                std::cerr << "Mismatch at (" << i << ", " << j << "): "
                    << value_result << " != " << value_output << std::endl;
                return false;
            }
        }
    }
    return true;
}

int main(int argc, char* argv[])
{
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <case_number>" << std::endl;
        return 1;
    }

    int case_number = std::atoi(argv[1]);
    if (case_number < 0 || case_number > 9) {
        std::cerr << "Case number must be between 0 and 9" << std::endl;
        return 1;
    }

    std::cout << "Test Case: " << case_number << std::endl;

    // Construct file paths
    std::string folder = "data/" + std::to_string(case_number) + "/";
    std::string input0_file = folder + "input0.raw";
    std::string input1_file = folder + "input1.raw";
    std::string result_file = folder + "result.raw";
    std::string reference_file = folder + "output.raw";

    int m = 0, n = 0, p = 0;
    std::string firstLine;

    // Read dimensions from input0.raw
    std::ifstream input0(input0_file);
    std::ifstream input1(input1_file);

    // Check if files opened successfully
    if (!input0.is_open() || !input1.is_open()) {
        std::cerr << "Dimension Extraction: Error Opening File" << std::endl;
        return 1;
    }

    if (std::getline(input0, firstLine)) {
        std::stringstream ss(firstLine);
        std::string part1, part2;
        ss >> part1 >> part2;
        m = std::stoi(part1);
        n = std::stoi(part2);
    }

    // Read dimensions from input1.raw
    if (std::getline(input1, firstLine)) {
        std::stringstream ss(firstLine);
        std::string part1, part2;
        ss >> part1 >> part2;
        if (std::stoi(part1) != n)
        {
            std::cerr << "Dimension Extraction: Matrix Dimensions Do Not Match" << std::endl;
        }
        p = std::stoi(part2);
    }

    // Allocate memory for matrices A, B, and C using new or malloc
    float* A = (float*)malloc(m * n * sizeof(float));
    float* B = (float*)malloc(n * p * sizeof(float));

    std::string line;

    // Read matrix A (row-major order)
    for (int i = 0; i < m; i++)
    {
        std::getline(input0, line);

        std::stringstream ss(line);
        for (int j = 0; j < n; j++)
        {
            ss >> A[i * n + j];
        }
    }

    // Read matrix B (row-major order)
    for (int i = 0; i < n; i++)
    {
        std::getline(input1, line);
        std::stringstream ss(line);
        for (int j = 0; j < p; j++)
        {
            ss >> B[i * p + j];
        }
    }

    float* C_cuda_naive = (float*)malloc(m * p * sizeof(float));
    float* C_cuda_tile = (float*)malloc(m * p * sizeof(float));

    // TODO: Use cudaMalloc and cudaMemcpy for GPU memory

    // Device Pointers
    float* A_d, * B_d, * C_d;

    // Calculate Sizes
    size_t size_A = m * n * sizeof(float);
    size_t size_B = n * p * sizeof(float);
    size_t size_C = m * p * sizeof(float);

    // Allocate Memory in GPU
    cudaError_t err;
    err = cudaMalloc((void**)&A_d, size_A);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for A_d: " << cudaGetErrorString(err) << std::endl;
    }

    err = cudaMalloc((void**)&B_d, size_B);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for B_d: " << cudaGetErrorString(err) << std::endl;
    }

    err = cudaMalloc((void**)&C_d, size_C);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for C_d: " << cudaGetErrorString(err) << std::endl;
    }

    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    cudaEventRecord(start_event, 0);

    // Copy Data from Host to GPU
    err = cudaMemcpy(A_d, A, size_A, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy A from host to device: " << cudaGetErrorString(err) << std::endl;
    }

    err = cudaMemcpy(B_d, B, size_B, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy B from host to device: " << cudaGetErrorString(err) << std::endl;
    }

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((p + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (m + threadsPerBlock.y - 1) / threadsPerBlock.y);

    naive_cuda_matmul << <numBlocks, threadsPerBlock >> > (C_d, A_d, B_d, m, n, p);

    err = cudaMemcpy(C_cuda_naive, C_d, size_C, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy C from device to host:" << cudaGetErrorString(err) << std::endl;
    }

    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);

    write_result(result_file, C_cuda_naive, m, p);
    bool naive_correct = validate_result(result_file, reference_file);
    if (!naive_correct) {
        std::cerr << "Naive CUDA failed for case " << case_number << std::endl;
    }
    else
    {
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start_event, stop_event);
        double seconds = milliseconds / 1000.0;
        std::cout << "Naive CUDA time: " << seconds << " s" << std::endl;
    }

    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    cudaEventRecord(start_event, 0);

    // Copy Data from Host to GPU
    err = cudaMemcpy(A_d, A, size_A, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy A from host to device: " << cudaGetErrorString(err) << std::endl;
    }

    err = cudaMemcpy(B_d, B, size_B, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy B from host to device: " << cudaGetErrorString(err) << std::endl;
    }

    dim3 threadsPerBlock_tiled(TILE_WIDTH, TILE_WIDTH);
    dim3 numBlocks_tiled((p + TILE_WIDTH - 1) / TILE_WIDTH, (m + TILE_WIDTH - 1) / TILE_WIDTH);

    tiled_cuda_matmul << <numBlocks_tiled, threadsPerBlock_tiled >> > (C_d, A_d, B_d, m, n, p, TILE_WIDTH);

    err = cudaMemcpy(C_cuda_tile, C_d, size_C, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to copy C from device to host:" << cudaGetErrorString(err) << std::endl;
    }

    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);

    write_result(result_file, C_cuda_tile, m, p);
    bool tiled_correct = validate_result(result_file, reference_file);
    if (!tiled_correct) {
        std::cerr << "Tiled CUDA failed for case " << case_number << std::endl;
    }
    else {
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start_event, stop_event);
        double seconds = milliseconds / 1000.0;
        std::cout << "Tiled CUDA time: " << seconds << " s" << std::endl;
    }

    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    // Clean up
    free(A);
    free(B);
    free(C_cuda_naive);
    free(C_cuda_tile);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    return 0;
}