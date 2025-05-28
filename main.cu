#include <iostream>
#include <fstream>
#include <string>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__global__ void naive_cuda_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p) {
    // Implement naive CUDA matrix multiplication
    uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < p) {
        float sum = 0.00f;

        for (uint32_t k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * p + col];
        }

        C[row * p + col] = sum;
    }
}

__global__ void tiled_cuda_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p, uint32_t tile_width) {
    // Implement tiled CUDA matrix multiplication
    __shared__ float a_shared[BLOCK_SIZE*BLOCK_SIZE];
    __shared__ float b_shared[BLOCK_SIZE*BLOCK_SIZE];

    uint32_t row = blockIdx.y*BLOCK_SIZE + threadIdx.y;
    uint32_t col = blockIdx.x*BLOCK_SIZE + threadIdx.x;

    float temp = 0.00f;

    for (uint32_t i = 0; i < (BLOCK_SIZE + n - 1)/BLOCK_SIZE; i++) {

        if (i * BLOCK_SIZE + threadIdx.x < n && row < m){
            a_shared[threadIdx.y * BLOCK_SIZE + threadIdx.x] = A[(row * n) + (i * BLOCK_SIZE) + threadIdx.x];
        } else {
            a_shared[threadIdx.y * BLOCK_SIZE + threadIdx.x] = 0.00f;
        }

        if (i * BLOCK_SIZE + threadIdx.y < n && col < p){
            b_shared[threadIdx.y * BLOCK_SIZE + threadIdx.x] = B[(i*BLOCK_SIZE + threadIdx.y)*p + col];
        } else{
            b_shared[threadIdx.y * BLOCK_SIZE + threadIdx.x] = 0.00f;
        }

        __syncthreads();

        for (uint32_t j = 0; j < BLOCK_SIZE; ++j){
            temp += a_shared[threadIdx.y * BLOCK_SIZE + j] * b_shared[j * BLOCK_SIZE + threadIdx.x];
        }
        __syncthreads();
    }

    if (row < m && col < p) {
        C[row * p + col] = temp;
    }
}

// Writing the files values into the matrix
void read_file(std::ifstream &input, float *matrix, uint32_t x, uint32_t y) {
    for (uint32_t j = 0; j < x; j++) {
        for (uint32_t k = 0; k < y; k++) {
            input >> matrix[j * y + k];
        }
    }
}

bool validate_result(const std::string &result_file, const std::string &reference_file) {
    // Implement result validation (same as Assignment 2)
    uint32_t x, y;
    float *R, *O;

    // Implement result validation
    std::ifstream result(result_file);
    if (result.is_open()) {
        result >> x >> y;
        R = new float[x * y];

        read_file(result, R, x, y);
    } else {
        exit(1);
    }

    std::ifstream output(reference_file);
    if (output.is_open()) {
        output >> x >> y;
        O = new float[x * y];

        read_file(output, O, x, y);
    } else {
        exit(1);
    }

    const float EPSILON = 1e-2f;

    for (uint32_t j = 0; j < x; j++) {
        for (uint32_t k = 0; k < y; k++) {
            // Checking that floating point numbers are the same
            if (std::fabs(R[j * y + k] - O[j * y + k]) > EPSILON) {
                std::cout << R[j * y + k] << " " << O[j * y + k] << std::endl;
                delete[] R;
                delete[] O;
                return false;
            }
        }
    }

    result.close();
    output.close();
    delete[] R;
    delete[] O;
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
    std::string result_file_naive = folder + "result_naive.raw";
    std::string result_file_tiled = folder + "result_tiled.raw";
    std::string reference_file = folder + "output.raw";

    // Read input0.raw (matrix A) and input1.raw (matrix B)
    float *A, *B, *C_naive, *C_tiled;
    uint32_t m, n, p;

    std::ifstream input0(input0_file);
    if (input0.is_open()) {
        input0 >> m >> n;
        A = new float[m * n];

        read_file(input0, A, m, n);
    } else {
        exit(1);
    }

    input0.close();

    std::ifstream input1(input1_file);
    if (input1.is_open()) {
        input1 >> n >> p;
        B = new float[n * p];

        read_file(input1, B, n, p);
    } else {
        exit(1);
    }

    input1.close();

    C_naive = new float[m * p];
    C_tiled = new float[m * p];

    // TODO: Use cudaMalloc and cudaMemcpy for GPU memory
    float *d_A, *d_B, *d_C_naive, *d_C_tiled;
    cudaMalloc(&d_A, sizeof(float)*m*n);
    cudaMalloc(&d_B, sizeof(float)*n*p);
    cudaMalloc(&d_C_naive, sizeof(float)*m*p);
    cudaMalloc(&d_C_tiled, sizeof(float)*m*p);
    cudaMemcpy(d_A, A, sizeof(float)*m*n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeof(float)*n*p, cudaMemcpyHostToDevice);

    // Measure naive CUDA performance
    cudaEvent_t start_naive, stop_naive;
    cudaEventCreate(&start_naive);
    cudaEventCreate(&stop_naive);

    // TODO: Launch naive_cuda_matmul kernel
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((p + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEventRecord(start_naive);
    naive_cuda_matmul<<<gridDim, blockDim>>>(d_C_naive, d_A, d_B, m, n, p);
    cudaDeviceSynchronize();
    cudaEventRecord(stop_naive);

    cudaMemcpy(C_naive, d_C_naive, sizeof(float) * m * p, cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop_naive);

    float naive_cuda_time = 0;
    cudaEventElapsedTime(&naive_cuda_time, start_naive, stop_naive);
    naive_cuda_time = naive_cuda_time / 1000;

    // TODO: Write naive CUDA result to file and validate
    std::ofstream result_naive(result_file_naive);

    result_naive << m << " " << p << "\n";
    for (uint32_t j = 0; j < m; j++) {
        for (uint32_t k = 0; k < p; k++) {
            result_naive << C_naive[j * p + k];

            if (k != p - 1) {
                result_naive << " ";
            }
        }
        if (j != m - 1) {
            result_naive << "\n";
        }
    }
    result_naive.close();

    // Validate naive result
    bool naive_correct = validate_result(result_file_naive, reference_file);
    if (!naive_correct) {
        std::cerr << "Naive result validation failed for case " << case_number << std::endl;
    }

    // Measure tiled CUDA performance
    cudaEvent_t start_tiled, stop_tiled;
    cudaEventCreate(&start_tiled);
    cudaEventCreate(&stop_tiled);

    // TODO: Launch tiled_cuda_matmul kernel
    cudaEventRecord(start_tiled);
    tiled_cuda_matmul<<<gridDim, blockDim>>>(d_C_tiled, d_A, d_B, m, n, p, 16);
    cudaDeviceSynchronize();
    cudaEventRecord(stop_tiled);

    cudaMemcpy(C_tiled, d_C_tiled, sizeof(float) * m * p, cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop_tiled);

    float tiled_cuda_time = 0;
    cudaEventElapsedTime(&tiled_cuda_time, start_tiled, stop_tiled);
    tiled_cuda_time = tiled_cuda_time / 1000;
    // TODO: Write tiled CUDA result to file and validate
    std::ofstream result_tiled(result_file_tiled);

    result_tiled << m << " " << p << "\n";
    for (uint32_t j = 0; j < m; j++) {
        for (uint32_t k = 0; k < p; k++) {
            result_tiled << C_tiled[j * p + k];

            if (k != p - 1) {
                result_tiled << " ";
            }
        }
        if (j != m - 1) {
            result_tiled << "\n";
        }
    }
    result_tiled.close();

    // Validate naive result
    bool tiled_correct = validate_result(result_file_tiled, reference_file);
    if (!tiled_correct) {
        std::cerr << "Tiled result validation failed for case " << case_number << std::endl;
    }

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
    cudaFree(d_C_naive);
    cudaFree(d_C_tiled);

    return 0;
}