#include <iostream>
#include <fstream>
using namespace std;
#include <string>
#include <cuda_runtime.h>


#define TILE_WIDTH 16  // Try 16 and 32

__global__ void naive_cuda_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p) {
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

__global__ void tiled_cuda_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p, uint32_t tile_width) {
    __shared__ float tile_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tile_B[TILE_WIDTH][TILE_WIDTH];

    uint32_t row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    uint32_t col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    float value = 0.0f;

    for (int ph = 0; ph < (n + TILE_WIDTH - 1) / TILE_WIDTH; ++ph) {
        if (row < m && ph * TILE_WIDTH + threadIdx.x < n)
            tile_A[threadIdx.y][threadIdx.x] = A[row * n + ph * TILE_WIDTH + threadIdx.x];
        else
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < p && ph * TILE_WIDTH + threadIdx.y < n)
            tile_B[threadIdx.y][threadIdx.x] = B[(ph * TILE_WIDTH + threadIdx.y) * p + col];
        else
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k)
            value += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];

        __syncthreads();
    }

    if (row < m && col < p)
        C[row * p + col] = value;
}

bool validate_result(const std::string &result_file, const std::string &reference_file) {
    std::ifstream res(result_file);
    std::ifstream ref(reference_file);

    if (!res.is_open() || !ref.is_open()) {
        std::cerr << "Error opening result or reference file!" << std::endl;
        return false;
    }

    int res_rows, res_cols, ref_rows, ref_cols;
    res >> res_rows >> res_cols;
    ref >> ref_rows >> ref_cols;

    if (res_rows != ref_rows || res_cols != ref_cols) {
        std::cerr << "Dimension mismatch in validation!" << std::endl;
        return false;
    }

    float res_val, ref_val;
    bool is_valid = true;

    for (int i = 0; i < res_rows * res_cols; ++i) {
        res >> res_val;
        ref >> ref_val;
        if (res_val != ref_val) {
            std::cerr << "Mismatch at element " << i << ": " << res_val << " vs " << ref_val << std::endl;
            is_valid = false;
        }
    }

    res.close();
    ref.close();
    return is_valid;
}

bool write_matrix_to_file(const std::string& filename, float* matrix, int m, int p) {
    std::ofstream outFile(filename);
    if (!outFile.is_open()) {
        std::cerr << "Error writing to file: " << filename << std::endl;
        return false;
    }

    outFile << m << " " << p << "\n";
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) {
            outFile << matrix[i * p + j];
            if (j < p - 1) outFile << " ";
        }
        outFile << "\n";
    }
    return true;
}
int main(int argc, char *argv[]) {

    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <case_number>" << std::endl;
        return 1;
    }

    int case_number = std::atoi(argv[1]);
    if (case_number < 0 || case_number > 9)
    {
        std::cerr << "Case number must be between 0 and 9" << std::endl;
        return 1;
    }

    // TODO: Read input0.raw (matrix A) and input1.raw (matrix B)
    
    // Construct file paths
    std::string folder = "data/" + std::to_string(case_number) + "/";
    std::string input0_file = folder + "input0.raw";
    std::string input1_file = folder + "input1.raw";
    std::string reference_file = folder + "output.raw";
    std::string result_naive_file = folder + "result_naive_cuda.raw";
    std::string result_tiled_file = folder + "result_tiled_cuda.raw";

    int m, n, p; // A is m x n, B is n x p, C is m x p

    // TODO Read input0.raw (matrix A)
    ifstream fileA(input0_file);
    fileA >> m >> n;
    // TODO Read input1.raw (matrix B)
    ifstream fileB(input1_file);
    fileB >> n >> p;

    float *h_A = new float[m * n];
    float *h_B = new float[n * p];
    float *h_C_naive = new float[m * p];
    float *h_C_tiled = new float[m * p];

    for (int i = 0; i < m * n; ++i) fileA >> h_A[i];
    for (int i = 0; i < n * p; ++i) fileB >> h_B[i];

    fileA.close();
    fileB.close();
    // TODO: Use cudaMalloc and cudaMemcpy for GPU memory
    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, m * n * sizeof(float));
    cudaMalloc(&d_B, n * p * sizeof(float));
    cudaMalloc(&d_C, m * p * sizeof(float));


   
    // Measure naive CUDA performance
    cudaEvent_t start_naive, stop_naive;
    cudaEventCreate(&start_naive);
    cudaEventCreate(&stop_naive);

    cudaMemcpy(d_A, h_A, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n * p * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((p + 15) / 16, (m + 15) / 16);

    cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    std::cerr << "CUDA Error 1: " << cudaGetErrorString(err) << std::endl;
}
    cudaEventRecord(start_naive);
    naive_cuda_matmul<<<gridSize, blockSize>>>(d_C, d_A, d_B, m, n, p);
    cudaEventRecord(stop_naive);

    cudaMemcpy(h_C_naive, d_C, m * p * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop_naive);
    float naive_cuda_time;
    cudaEventElapsedTime(&naive_cuda_time, start_naive, stop_naive);
    naive_cuda_time /= 1000.0f;

    write_matrix_to_file(result_naive_file, h_C_naive, m, p);
bool valid_naive = validate_result(result_naive_file, reference_file);
    if (!valid_naive) std::cerr << "Naive CUDA validation failed." << std::endl;

    // --- Tiled ---
    cudaEvent_t start_tiled, stop_tiled;
    cudaEventCreate(&start_tiled); cudaEventCreate(&stop_tiled);

    cudaMemcpy(d_A, h_A, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n * p * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSizeTiled(TILE_WIDTH, TILE_WIDTH);
    dim3 gridSizeTiled((p + TILE_WIDTH - 1) / TILE_WIDTH, (m + TILE_WIDTH - 1) / TILE_WIDTH);

   err = cudaGetLastError();
if (err != cudaSuccess) {
    std::cerr << "CUDA Error 2: " << cudaGetErrorString(err) << std::endl;
}

    cudaEventRecord(start_tiled);
    tiled_cuda_matmul<<<gridSizeTiled, blockSizeTiled>>>(d_C, d_A, d_B, m, n, p, TILE_WIDTH);
    cudaEventRecord(stop_tiled);

    cudaMemcpy(h_C_tiled, d_C, m * p * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop_tiled);
    float tiled_cuda_time;

    cudaEventElapsedTime(&tiled_cuda_time, start_tiled, stop_tiled);  

    tiled_cuda_time /= 1000.0f;

    write_matrix_to_file(result_tiled_file, h_C_tiled, m, p);
bool valid_tiled = validate_result(result_tiled_file, reference_file);
    if (!valid_tiled) std::cerr << "Tiled CUDA validation failed." << std::endl;

    std::cout << "Case " << case_number << " (" << m << "x" << n << "x" << p << "):\n";
    std::cout << "Naive CUDA time: " << naive_cuda_time << " seconds\n";
    std::cout << "Tiled CUDA time: " << tiled_cuda_time << " seconds\n";
    std::cout << "Tiled CUDA Speedup: " << naive_cuda_time / tiled_cuda_time << "x\n";

    // Clean up
    delete[] h_A;
    delete[] h_B;
    delete[] h_C_naive;
    delete[] h_C_tiled;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}