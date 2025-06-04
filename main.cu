#include <iostream>
#include <fstream>
#include <string>
#include <cuda_runtime.h>

__global__ void naive_cuda_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < p) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * p + col];
        }
        C[row * p + col] = sum;
    }
}

__global__ void tiled_cuda_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p, uint32_t tile_width) {
    // TODO: Implement tiled CUDA matrix multiplication
}

bool validate_result(const std::string &result_file, const std::string &reference_file) {
    // TODO: Implement result validation (same as Assignment 2)
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

    std::string parentPath = getParentPath();
    std::cout << "Parent Path: " << parentPath << std::endl;

    std::string root_path = parentPath + "\\Homework-2\\data\\"; // Construct path relative to the parent directory
    std::cout << "Root Path: " << root_path << std::endl;


    // Construct file paths
    std::string folder = std::to_string(case_number) + "\\";
    std::string input0_file = root_path + folder + "input0.raw";
    std::string input1_file = root_path + folder + "input1.raw";
    std::string result_file = root_path + folder + "result.raw";
    std::string reference_file = root_path + folder + "output.raw";

    std::ifstream ifs;


    std::string input_dir_1 = root_path + folder + "input0.raw";
    std::string input_dir_2 = root_path + folder + "input1.raw";
    std::string result_dir = root_path + folder + "result.raw";

    std::cout << "input0 path dir: " << input_dir_1 << std::endl;

    std::ifstream input0(input_dir_1);
    std::ifstream input1(input_dir_2);
    if (!input0.is_open() || !input1.is_open()) {
        std::cerr << "Error opening input files." << std::endl;
        return -1;
    }

    int m, n;
    input0 >> m;
    input0 >> n;

    float *A = (float *)malloc(m * n * sizeof(float));
    if (!A) {
        std::cerr << "Memory allocation failed for matrix A." << std::endl;
        return -1;
    }

    // TODO Read input0.raw (matrix A)
    if (!input0.is_open())
    {
        std::cerr << "Error opening input0.raw" << std::endl;
        return -1;
    }

    for (int i = 0; i < m * n; ++i) input0 >> A[i];
    input0.close();

    int n_check, p;
    input1 >> n_check;
    input1 >> p;


    // TODO Read input1.raw (matrix B)
    if (!input1.is_open()) {
        std::cerr << "Error opening input1.raw" << std::endl;
        delete[] A;
        return -1;
    }
    float *B = new float[n * p];

    for (int i = 0; i < n * p; ++i) input1 >> B[i];
    input1.close();


    // Allocate memory for result matrices
    float *C_naive = new float[m * p];

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, m * n * sizeof(float));
    cudaMalloc(&d_B, n * p * sizeof(float));
    cudaMalloc(&d_C, m * p * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_A, A, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, n * p * sizeof(float), cudaMemcpyHostToDevice);

    // Kernel launch configuration
    dim3 blockDim(16, 16);
    dim3 gridDim((p + blockDim.x - 1) / blockDim.x, (m + blockDim.y - 1) / blockDim.y);

    // Launch kernel
    naive_cuda_matmul << <gridDim, blockDim >> > (d_C, d_A, d_B, m, n, p);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(C_naive, d_C, m * p * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);





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
    delete[] A;
    free(A);
    delete[] B;
    delete[] C_naive;

    return 0;
}



