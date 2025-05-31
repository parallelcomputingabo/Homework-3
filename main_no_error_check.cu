#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda_runtime.h>

#include <sstream>
#include <iomanip>

#include <fstream>
#include <cmath>
#include <vector>
#include <string>


//-------------------------------------------------------------------------------------------------
// CUDA kernel for naive matrix multiplication: each thread computes one output element C[row, col]
//-------------------------------------------------------------------------------------------------
__global__ void naive_cuda_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p) {
    // Compute global row and column indices
    uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;

    // Only proceed if within matrix bounds
    if (row < m && col < p) {
        float sum = 0.0f;
        // Dot product over k
        for (uint32_t k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * p + col];
        }
        C[row * p + col] = sum;
    }
}


//-------------------------------------------------------------------------------------------------
// CUDA kernel for tiled matrix multiplication using shared memory.
// Each block loads a TILE×TILE submatrix of A and B into shared memory.
//-------------------------------------------------------------------------------------------------
__global__ void tiled_cuda_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p, uint32_t tile_width) {
    // Declare shared memory: first tile_width^2 floats for A, next tile_width^2 for B
    extern __shared__ float shared_mem[];
    float* As = shared_mem;
    float* Bs = &shared_mem[tile_width * tile_width];

    // Compute global row and column
    uint32_t row = blockIdx.y * tile_width + threadIdx.y;
    uint32_t col = blockIdx.x * tile_width + threadIdx.x;

    float value = 0.0f;  // Accumulator for this thread's output

    // Loop over all tiles
    for (uint32_t t = 0; t < (n + tile_width - 1) / tile_width; ++t) {
        // Load element of A into shared memory if within bounds, else 0
        if (row < m && t * tile_width + threadIdx.x < n)
            As[threadIdx.y * tile_width + threadIdx.x] = A[row * n + t * tile_width + threadIdx.x];
        else
            As[threadIdx.y * tile_width + threadIdx.x] = 0.0f;

        // Load element of B into shared memory if within bounds, else 0
        if (col < p && t * tile_width + threadIdx.y < n)
            Bs[threadIdx.y * tile_width + threadIdx.x] = B[(t * tile_width + threadIdx.y) * p + col];
        else
            Bs[threadIdx.y * tile_width + threadIdx.x] = 0.0f;

        __syncthreads();  // Ensure all threads have loaded their tile elements

        // Multiply the two tiles
        for (uint32_t k = 0; k < tile_width; ++k)
            value += As[threadIdx.y * tile_width + k] * Bs[k * tile_width + threadIdx.x];

        __syncthreads();  // Ensure all threads have finished computing before next tile load
    }

    // Write the result if within bounds
    if (row < m && col < p)
        C[row * p + col] = value;
}


//-------------------------------------------------------------------------------------------------
// Load a matrix from a .raw file: first line contains "rows cols", followed by float values.
// Returns a pointer to the allocated 1D float array (row-major).
//-------------------------------------------------------------------------------------------------
float* load_matrix_with_dims(const char *filename, uint32_t *rows, uint32_t *cols) {
    FILE *f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "Error opening file %s\n", filename);
        exit(EXIT_FAILURE);
    }

    // Read dimensions from the first line
    if (fscanf(f, "%u %u\n", rows, cols) != 2) {
        fprintf(stderr, "Error reading dimensions from %s\n", filename);
        exit(EXIT_FAILURE);
    }

    size_t count = (*rows) * (*cols);
    float *matrix = (float*) malloc(count * sizeof(float));

    // Read the matrix values
    for (size_t i = 0; i < count; ++i) {
        if (fscanf(f, "%f", &matrix[i]) != 1) {
            fprintf(stderr, "Error reading float values from %s\n", filename);
            exit(EXIT_FAILURE);
        }
    }

    fclose(f);
    return matrix;
}


//-------------------------------------------------------------------------------------------------
// Save a matrix to a .raw file: first line "rows cols", then each element formatted to strip trailing zeros.
//-------------------------------------------------------------------------------------------------
void save_matrix(const char *filename, float *matrix, uint32_t m, uint32_t p) {
    FILE *f = fopen(filename, "w");
    if (!f) {
        fprintf(stderr, "Error opening file %s\n", filename);
        exit(EXIT_FAILURE);
    }

    // Write dimensions
    fprintf(f, "%u %u\n", m, p);

    // Write values, rounding to two decimals and stripping trailing zeros
    for (uint32_t i = 0; i < m; ++i) {
        for (uint32_t j = 0; j < p; ++j) {
            float v = std::round(matrix[i * p + j] * 100.0f) / 100.0f;
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(2) << v;
            std::string s = ss.str();
            while (s.back() == '0') s.pop_back();  // Remove trailing zeros

            fprintf(f, "%s%s", s.c_str(), (j + 1 < p ? " " : "\n"));
        }
    }

    fclose(f);
}


//-------------------------------------------------------------------------------------------------
// Compare two floats with tolerance ±0.001
//-------------------------------------------------------------------------------------------------
bool compare_floats(float a, float b) {
    return std::fabs(a - b) <= 0.001f;
}


//-------------------------------------------------------------------------------------------------
// Compare two matrix files line by line; return true if all floats match within tolerance.
//-------------------------------------------------------------------------------------------------
bool compare_matrix_files(const std::string& p1, const std::string& p2) {
    std::ifstream f1(p1), f2(p2);
    if (!f1 || !f2) return false;

    std::string l1, l2;
    while (std::getline(f1, l1)) {
        if (!std::getline(f2, l2)) return false;
        if (l1 == l2) continue;

        std::istringstream s1(l1), s2(l2);
        std::vector<float> v1, v2;
        float x;
        while (s1 >> x) v1.push_back(x);
        while (s2 >> x) v2.push_back(x);

        if (v1.size() != v2.size()) return false;
        for (size_t i = 0; i < v1.size(); ++i)
            if (!compare_floats(v1[i], v2[i])) return false;
    }

    return f1.eof() && !std::getline(f2, l2);
}


//-------------------------------------------------------------------------------------------------
// Main function: runs naive and tiled CUDA matrix multiplication, measures times, and validates.
//-------------------------------------------------------------------------------------------------
int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <inputA> <inputB> <outputC>\n", argv[0]);
        return 1;
    }

    // Create CUDA events for total timing (H2D + kernel + D2H) for naive version
    cudaEvent_t totalStart, totalStop;
    cudaEventCreate(&totalStart);
    cudaEventCreate(&totalStop);

    const char *fileA = argv[1];
    const char *fileB = argv[2];
    const char *fileC = argv[3];

    // Load input matrices from files
    uint32_t m, n, n2, p;
    float *h_A = load_matrix_with_dims(fileA, &m, &n);  // A: m × n
    float *h_B = load_matrix_with_dims(fileB, &n2, &p); // B: n × p

    if (n != n2) {
        fprintf(stderr, "Matrix dimension mismatch: A columns (%u) != B rows (%u)\n", n, n2);
        exit(EXIT_FAILURE);
    }

    size_t size_A = m * n;
    size_t size_B = n * p;
    size_t size_C = m * p;

    // Allocate host memory for result
    float *h_C = (float *)malloc(size_C * sizeof(float));

    // Allocate device memory for A, B, C (naive)
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A * sizeof(float));
    cudaMalloc(&d_B, size_B * sizeof(float));
    cudaMalloc(&d_C, size_C * sizeof(float));

    // Start total timer for naive (include H2D)
    cudaEventRecord(totalStart);

    // Copy inputs to GPU
    cudaMemcpy(d_A, h_A, size_A * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B * sizeof(float), cudaMemcpyHostToDevice);

    // Launch configuration for naive
    dim3 blockDim(16, 16);
    dim3 gridDim((p + blockDim.x - 1) / blockDim.x,
                 (m + blockDim.y - 1) / blockDim.y);

    // Create events and measure kernel-only time for naive
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    naive_cuda_matmul<<<gridDim, blockDim>>>(d_C, d_A, d_B, m, n, p);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Naive CUDA time (kernel execution only): %.4f ms\n", ms);

    // Copy result back and stop total timer
    cudaMemcpy(h_C, d_C, size_C * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(totalStop);
    cudaEventSynchronize(totalStop);

    float totalMs = 0;
    cudaEventElapsedTime(&totalMs, totalStart, totalStop);
    printf("Total CUDA time (with transfers): %.4f ms\n", totalMs);

    // Save naive result and validate
    save_matrix(fileC, h_C, m, p);
    std::string ref_path = std::string(fileC).substr(0, std::string(fileC).find_last_of("/\\") + 1) + "output.raw";
    bool ok = compare_matrix_files(fileC, ref_path);
    printf("%s\n", ok ? "CUDA RESULT PASS" : "CUDA RESULT FAIL");


    //-------------------------------------------------------------------------------------------------
    // Tiled CUDA Matrix Multiplication
    //-------------------------------------------------------------------------------------------------
    const uint32_t TILE_WIDTH = 16;
    dim3 tiledBlockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 tiledGridDim((p + TILE_WIDTH - 1) / TILE_WIDTH,
                      (m + TILE_WIDTH - 1) / TILE_WIDTH);

    // Allocate device and host memory for tiled result
    float *d_C_tiled;
    cudaMalloc(&d_C_tiled, size_C * sizeof(float));
    float *h_C_tiled = (float*)malloc(size_C * sizeof(float));

    // Create CUDA events for total timing (H2D + kernel + D2H) for tiled
    cudaEvent_t tiledTotalStart, tiledTotalStop;
    cudaEventCreate(&tiledTotalStart);
    cudaEventCreate(&tiledTotalStop);

    // Start total timer for tiled
    cudaEventRecord(tiledTotalStart);

    // Re-upload inputs to include H2D in tiled total timing
    cudaMemcpy(d_A, h_A, size_A * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B * sizeof(float), cudaMemcpyHostToDevice);

    // Create events and measure kernel-only time for tiled
    cudaEvent_t tileStart, tileStop;
    cudaEventCreate(&tileStart);
    cudaEventCreate(&tileStop);
    cudaEventRecord(tileStart);

    tiled_cuda_matmul<<<tiledGridDim, tiledBlockDim, 2 * TILE_WIDTH * TILE_WIDTH * sizeof(float)>>>(
        d_C_tiled, d_A, d_B, m, n, p, TILE_WIDTH);
    cudaDeviceSynchronize();

    cudaEventRecord(tileStop);
    cudaEventSynchronize(tileStop);

    float tiledMs = 0;
    cudaEventElapsedTime(&tiledMs, tileStart, tileStop);
    printf("Tiled CUDA kernel time: %.4f ms\n", tiledMs);

    // Copy tiled result back and stop total timer
    cudaMemcpy(h_C_tiled, d_C_tiled, size_C * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(tiledTotalStop);
    cudaEventSynchronize(tiledTotalStop);

    float tiledTotalMs = 0;
    cudaEventElapsedTime(&tiledTotalMs, tiledTotalStart, tiledTotalStop);
    printf("Tiled CUDA total time (with transfers): %.4f ms\n", tiledTotalMs);

    // Save tiled result and validate
    std::string tiled_output_path = std::string(fileC).substr(0, std::string(fileC).find_last_of("/\\") + 1) + "result_tiled.raw";
    save_matrix(tiled_output_path.c_str(), h_C_tiled, m, p);

    std::string tiled_ref_path = std::string(fileC).substr(0, std::string(fileC).find_last_of("/\\") + 1) + "output.raw";
    bool ok_tiled = compare_matrix_files(tiled_output_path, tiled_ref_path);
    printf("%s\n", ok_tiled ? "TILED RESULT PASS" : "TILED RESULT FAIL");

    // Free tiled resources
    cudaFree(d_C_tiled);
    free(h_C_tiled);

    // Free naive resources
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
