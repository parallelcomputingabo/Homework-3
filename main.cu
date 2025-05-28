#include <iostream>
#include <fstream>
#include <string>
#include <cuda_runtime.h>

float *read_matrix(const std::string &filename, int &rows, int &cols)
{
    std::ifstream file(filename);
    if (!file)
    {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    file >> rows >> cols;
    float *mat = new float[rows * cols];

    for (int i = 0; i < rows * cols; ++i)
    {
        file >> mat[i];
    }

    file.close();
    return mat;
}

void write_matrix(const std::string &filename, float *mat, int rows, int cols)
{
    std::ofstream file(filename);
    if (!file)
    {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }
    file << rows << " " << cols << "\n";
    for (int i = 0; i < rows * cols; ++i)
    {
        file << mat[i] << " ";
        if ((i + 1) % cols == 0)
        {
            file << "\n";
        }
    }

    file.close();
}

__global__ void naive_cuda_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;


    float sum = 0.0f;
    if (row < m && col < p) {
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * p + col];
        }
        C[row * p + col] = sum;
    }
}

__global__ void tiled_cuda_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p, uint32_t tile_width) {
    __shared__ float tile_A[16][16];
    __shared__ float tile_B[16][16];

    int row = blockIdx.y * tile_width + threadIdx.y;
    int col = blockIdx.x * tile_width + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (n + tile_width - 1) / tile_width; ++t) {
        if (row < m && (t * tile_width + threadIdx.x) < n)
            tile_A[threadIdx.y][threadIdx.x] = A[row * n + t * tile_width + threadIdx.x];
        else
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < p && (t * tile_width + threadIdx.y) < n)
            tile_B[threadIdx.y][threadIdx.x] = B[(t * tile_width + threadIdx.y) * p + col];
        else
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int i = 0; i < tile_width; ++i) {
            sum += tile_A[threadIdx.y][i] * tile_B[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < m && col < p) {
        C[row * p + col] = sum;
    }


}

bool validate_result(const std::string &result_file, const std::string &reference_file) {
    int result_rows, result_cols, ref_rows, ref_cols;
    float *result = read_matrix(result_file, result_rows, result_cols);
    float *reference = read_matrix(reference_file, ref_rows, ref_cols);

    const float EPSILON = 1e-4f;

    for (int i = 0; i < result_rows * result_cols; ++i)
    {
        if (std::abs(result[i] - reference[i]) > EPSILON)
        {
            return false;
        }
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

    int m, n, p;
    float *A = read_matrix(input0_file, m, n);
    float *B = read_matrix(input1_file, n, p);
    float *C_naive = new float[m * p];
    float *C_tiled = new float[m * p];

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeof(float) * m * n);
    cudaMalloc(&d_B, sizeof(float) * n * p);
    cudaMalloc(&d_C, sizeof(float) * m * p);

    cudaEvent_t start_naive, stop_naive;
    cudaEventCreate(&start_naive);
    cudaEventCreate(&stop_naive);
    cudaEventRecord(start_naive);

    cudaMemcpy(d_A, A, sizeof(float) * m * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeof(float) * n * p, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((p + 15) / 16, (m + 15) / 16);

    // Launch naive kernel
    cudaMemset(d_C, 0, sizeof(float) * m * p);
    
    naive_cuda_matmul<<<gridDim, blockDim>>>(d_C, d_A, d_B, m, n, p);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize(); 

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel execution error: %s\n", cudaGetErrorString(err));
    }

    cudaMemcpy(C_naive, d_C, sizeof(float) * m * p, cudaMemcpyDeviceToHost);
    
    cudaEventRecord(stop_naive);
    cudaEventSynchronize(stop_naive);
    float naive_cuda_time;
    cudaEventElapsedTime(&naive_cuda_time, start_naive, stop_naive);
    write_matrix(result_file, C_naive, m, p);
    validate_result(result_file, reference_file);

    //Launch tiled cuda kernel
    int tile_width = 16;

    dim3 tileBlock(tile_width, tile_width);
    dim3 tileGrid((p + tile_width - 1) / tile_width, (m + tile_width - 1) / tile_width);

    cudaEvent_t start_tiled, stop_tiled;
    cudaEventCreate(&start_tiled);
    cudaEventCreate(&stop_tiled);

    cudaEventRecord(start_tiled);

    cudaMemcpy(d_A, A, sizeof(float) * m * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeof(float) * n * p, cudaMemcpyHostToDevice);

    cudaMemset(d_C, 0, sizeof(float) * m * p);
    tiled_cuda_matmul<<<tileGrid, tileBlock>>>(d_C, d_A, d_B, m, n, p, tile_width);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel execution error: %s\n", cudaGetErrorString(err));
    }
    cudaMemcpy(C_tiled, d_C, sizeof(float) * m * p, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop_tiled);
    cudaEventSynchronize(stop_tiled);
    
    float tiled_cuda_time;
    cudaEventElapsedTime(&tiled_cuda_time, start_tiled, stop_tiled);
    write_matrix(result_file, C_tiled, m, p);
    validate_result(result_file, reference_file);
    


    // Print performance results
    std::cout << "Case " << case_number << " (" << m << "x" << n << "x" << p << "):\n";
    std::cout << "Naive CUDA time: " << naive_cuda_time << " seconds\n";
    std::cout << "Tiled CUDA time: " << tiled_cuda_time << " seconds\n";

    // Clean up
    delete[] A;
    delete[] B;
    delete[] C_naive;
    delete[] C_tiled;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}