#include <iostream>
#include <fstream>
#include <string>
#include <cuda_runtime.h>
#include <cmath>
#include <iomanip>

// define tile width 
#define TILE_WIDTH 16

__global__ void naive_cuda_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p) {
    // TODO: Implement naive CUDA matrix multiplication
    // row col indices for thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < p) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            // dot product
            sum += A[row * n + k] * B[k * p + col];
        }
        // store in matrix C
        C[row * p + col] = sum;
    }
}

__global__ void tiled_cuda_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p, uint32_t tile_width) {
    // TODO: Implement tiled CUDA matrix multiplication
    __shared__ float tile_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tile_B[TILE_WIDTH][TILE_WIDTH];

    // thread indices
    int row = blockIdx.y * tile_width + threadIdx.y;
    int col = blockIdx.x * tile_width + threadIdx.x;

    float sum = 0.0f;

    // looping over the tiles A and B
    for (int t = 0; t < (n + tile_width -1)/tile_width; ++t){
        // load tile of A into shared mem
        if (row < m && (t * tile_width + threadIdx.x) < n){
            tile_A[threadIdx.y][threadIdx.x] = A[row * n + t * tile_width + threadIdx.x];
            }
        else{
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
        }
        // load B to shared mem
        if (col < p && (t * tile_width + threadIdx.y) < n){
            tile_B[threadIdx.y][threadIdx.x] = B[(t * tile_width + threadIdx.y) * p + col];
        }
        else {
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // wait for threads to finish with loading tiles
        __syncthreads();

        // multiplication on tile level
        for (int k = 0; k < tile_width; ++k){
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }

        __syncthreads();
    }
    if (row < m && col < p){
        // result in C
        C[row * p + col] = sum;
    }

}

bool validate_result(const std::string &result_file, const std::string &reference_file) {
    // TODO: Implement result validation (same as Assignment 2)
       //TODO : Implement result validation
   // input is file path to result.raw and output.raw
    // read files line for line and compare
    // return true or false
    std::ifstream resultFile(result_file);
    std::ifstream outputFile(reference_file);

    int res_m, res_p, out_m, out_p;

    resultFile >> res_m >> res_p;
    outputFile >> out_m >> out_p;

    for (uint32_t i = 0; i < static_cast<uint32_t>(res_m); i++){
        for (uint32_t j = 0; j < static_cast<uint32_t>(res_p); j++){
            float result_val, expected_val;
            resultFile >> result_val;
            outputFile >> expected_val;

            //adding a small error tolerance for floating point because sometimes its not exact (421.51, 421.5)
            // const float error_tol = 0.000001f;

            if (result_val != expected_val){
                std::cerr << "error at " << i << ", " << j << ". Expected " << expected_val << ", got " << result_val << std::endl;
                return false;
            }

        }
    }
    std::cout << "Matrix is same, validation complete\n";
    return true;
}

// for writing floats
void write_dot_after_value(std::ofstream& file, float value) {
    if (std::floor(value) == value) {
        file << std::fixed << std::setprecision(1) << value;
    } else {
        file << std::fixed << std::setprecision(2) << value;
    }
}

void write_matrix_to_file(const std::string &filename, float* C, uint32_t m, uint32_t p){
    std::ofstream result(filename);

    result << m << " " << p << "\n";
    for (uint32_t i = 0; i < m; ++i){
        for (uint32_t j = 0; j < p; ++j){
            write_dot_after_value(result, C[i * p + j]);
            if (j < p - 1){
                result << " ";
            }
        }
        result << "\n";
    }
    result.flush();
    result.close();
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
    // should be ./data now because of mahti
    std::string folder = "./data/" + std::to_string(case_number) + "/";
    std::string input0_file = folder + "input0.raw";
    std::string input1_file = folder + "input1.raw";
    std::string result_file_naive = folder + "result_naive.raw";
    std::string result_file_tiled = folder + "result_tiled.raw";
    std::string reference_file = folder + "output.raw";

    int m, n, p;

    // TODO: Read input0.raw (matrix A) and input1.raw (matrix B)

    // matrix A
    // TODO Read input0.raw (matrix A)
    std::ifstream input0(input0_file);
    input0 >> m >> n;

    // TODO Read input1.raw (matrix B)
    std::ifstream input1(input1_file);
    input1 >> n >> p;

    float* A = new float[m * n];
    float* B = new float[n * p];


    // Read elements to A and B
    // A matrix
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            input0 >> A[i* n + j];
        }
    }
    // B matrix
    for (int i = 0; i<n; i++){
        for (int j = 0; j < p; j++){
            input1 >> B[i*p + j];
        }
    }

    input0.close();
    input1.close();


    // TODO: Use cudaMalloc and cudaMemcpy for GPU memory
    // allocates device memory
    float *d_A, *d_B, *d_C;


    cudaMalloc(&d_A, sizeof(float) * m * n);
    cudaMalloc(&d_B, sizeof(float) * n * p);
    cudaMalloc(&d_C, sizeof(float) * m * p);

    cudaMemcpy(d_A, A, sizeof(float) * m * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeof(float) * n * p, cudaMemcpyHostToDevice);

    // Measure naive CUDA performance
    // TODO: Launch naive_cuda_matmul kernel
    // block and grid layout for kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((p + blockSize.x - 1)/ blockSize.x, (m + blockSize.y - 1) / blockSize.y);

    // time of events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // naive matmul kernel start
    cudaEventRecord(start);
    naive_cuda_matmul<<<gridSize, blockSize>>>(d_C, d_A, d_B, m, n, p);

    // check for error after kernel start
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error : " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float millisec = 0;

    cudaEventElapsedTime(&millisec, start, stop);
    
    float naive_cuda_time = millisec / 1000.0f;

    float* C_naive = new float[m * p];
    cudaMemcpy(C_naive, d_C, sizeof(float) * m * p, cudaMemcpyDeviceToHost);

    // TODO: Write naive CUDA result to file and validate
    write_matrix_to_file(result_file_naive, C_naive, m, p);
    bool naive_correct = validate_result(result_file_naive, reference_file);
    if (!naive_correct) {
        std::cerr << "Naive result validation failed for case " << case_number << std::endl;
    }




    // Measure tiled CUDA performance

    //grid for tiled matmul
    dim3 tiledBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 tiledGrid((p + TILE_WIDTH - 1) / TILE_WIDTH, (m + TILE_WIDTH - 1) / TILE_WIDTH);

    cudaEventRecord(start);
    // tiled kernel start
    tiled_cuda_matmul<<<tiledGrid, tiledBlock>>>(d_C, d_A, d_B, m, n, p, TILE_WIDTH);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error : " << cudaGetErrorString(err) << std::endl;
        return 1;
    }


    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&millisec, start, stop);
    float tiled_cuda_time = millisec / 1000.0f;

    float* C_tiled = new float[m * p];
    cudaMemcpy(C_tiled, d_C, sizeof(float) * m * p, cudaMemcpyDeviceToHost);

    // TODO: Launch tiled_cuda_matmul kernel

    // TODO: Write tiled CUDA result to file and validate

    write_matrix_to_file(result_file_tiled, C_tiled, m, p);
    bool tiled_correct = validate_result(result_file_tiled, reference_file);
    if (!tiled_correct) {
        std::cerr << "tiled result validation failed for case " << case_number << std::endl;
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
    cudaFree(d_C);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}