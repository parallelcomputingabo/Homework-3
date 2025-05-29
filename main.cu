#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                    \
    {                                                                       \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__   \
                      << " code=" << err << " ("                          \
                      << cudaGetErrorString(err) << ")" << std::endl;     \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                   \
    }

// Read a matrix: first two ints are rows, cols, then the whole matrix floats
bool read_matrix(const std::string& filename, float*& mat,
                 uint32_t& rows, uint32_t& cols) {
    std::ifstream in(filename);
    if (!in) {
        std::cerr << "Cannot open file " << filename << std::endl;
        return false;
    }
    in >> rows >> cols;
    mat = (float*)std::malloc((size_t)rows * cols * sizeof(float));
    if (!mat) {
        std::cerr << "Host memory allocation failed" << std::endl;
        return false;
    }
    uint64_t total = (uint64_t)rows * cols;
    for (uint64_t i = 0; i < total; ++i) {
        in >> mat[i];
    }
    return true;
}

// Write rows cols, then the whole matrix floats
bool write_matrix(const std::string& filename,
                  const float* mat, uint32_t rows, uint32_t cols) {
    std::ofstream out(filename);
    if (!out) {
        std::cerr << "Cannot write to file " << filename << std::endl;
        return false;
    }
    out << rows << " " << cols << "\n"
        << std::fixed << std::setprecision(2);
    uint64_t total = (uint64_t)rows * cols;
    for (uint64_t i = 0; i < total; ++i) {
        out << mat[i]
            << ((i % cols == cols - 1) ? "\n" : " ");
    }
    return true;
}

// Validate result by comparing to reference within epsilon
bool validate_result(const std::string &result_file,
                     const std::string &reference_file) {
    float *C1 = nullptr, *C2 = nullptr;
    uint32_t r1, c1, r2, c2;
    if (!read_matrix(result_file, C1, r1, c1)) return false;
    if (!read_matrix(reference_file, C2, r2, c2)) {
        std::free(C1);
        return false;
    }
    if (r1 != r2 || c1 != c2) {
        std::free(C1);
        std::free(C2);
        return false;
    }
    const float EPS = 1e-2f; // match two-decimal precision
    uint64_t total = (uint64_t)r1 * c1;
    bool ok = true;
    for (uint64_t i = 0; i < total; ++i) {
        if (std::fabs(C1[i] - C2[i]) > EPS) {
            ok = false;
            break;
        }
    }
    std::free(C1);
    std::free(C2);
    return ok;
}

// Naive CUDA matrix multiplication kernel: one thread per element of C
__global__ void naive_cuda_matmul(float *C, const float *A, const float *B,
                                  uint32_t m, uint32_t n, uint32_t p) {
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

// Tiled CUDA matrix multiplication kernel using shared memory
__global__ void tiled_cuda_matmul(float *C, const float *A, const float *B,
                                  uint32_t m, uint32_t n, uint32_t p,
                                  uint32_t tile_width) {
    extern __shared__ float shared_mem[];
    float *As = shared_mem;
    float *Bs = shared_mem + tile_width * tile_width;
    uint32_t row = blockIdx.y * tile_width + threadIdx.y;
    uint32_t col = blockIdx.x * tile_width + threadIdx.x;
    float value = 0.0f;
    uint32_t num_tiles = (n + tile_width - 1) / tile_width;
    for (uint32_t t = 0; t < num_tiles; ++t) {
        uint32_t A_col = t * tile_width + threadIdx.x;
        uint32_t B_row = t * tile_width + threadIdx.y;
        // Load A tile
        if (row < m && A_col < n)
            As[threadIdx.y * tile_width + threadIdx.x] = A[row * n + A_col];
        else
            As[threadIdx.y * tile_width + threadIdx.x] = 0.0f;
        // Load B tile
        if (B_row < n && col < p)
            Bs[threadIdx.y * tile_width + threadIdx.x] = B[B_row * p + col];
        else
            Bs[threadIdx.y * tile_width + threadIdx.x] = 0.0f;
        __syncthreads();
        // Compute partial
        for (uint32_t k = 0; k < tile_width; ++k) {
            value += As[threadIdx.y * tile_width + k] *
                     Bs[k * tile_width + threadIdx.x];
        }
        __syncthreads();
    }
    if (row < m && col < p) {
        C[row * p + col] = value;
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <case_number>\n";
        return EXIT_FAILURE;
    }
    int case_number = std::atoi(argv[1]);
    if (case_number < 0 || case_number > 9) {
        std::cerr << "Case number must be between 0 and 9\n";
        return EXIT_FAILURE;
    }
    std::string folder = "data/" + std::to_string(case_number) + "/";
    std::string inputA = folder + "input0.raw";
    std::string inputB = folder + "input1.raw";
    std::string reference = folder + "output.raw";
    std::string result_naive = folder + "result_naive.raw";
    std::string result_tiled = folder + "result_tiled.raw";

    // Read input matrices
    float *h_A = nullptr, *h_B = nullptr;
    uint32_t m, n, n2, p;
    if (!read_matrix(inputA, h_A, m, n) ||
        !read_matrix(inputB, h_B, n2, p) || n2 != n) {
        std::cerr << "Failed to read inputs or dimension mismatch\n";
        return EXIT_FAILURE;
    }
    // Allocate host outputs
    float *h_C_naive = (float*)std::malloc((size_t)m * p * sizeof(float));
    float *h_C_tiled = (float*)std::malloc((size_t)m * p * sizeof(float));
    if (!h_C_naive || !h_C_tiled) {
        std::cerr << "Host output allocation failed\n";
        return EXIT_FAILURE;
    }
    // Allocate device memory
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, (size_t)m * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, (size_t)n * p * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, (size_t)m * p * sizeof(float)));

    // Setup CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // ----- Naive CUDA -----
    dim3 block_naive(16, 16);
    dim3 grid_naive((p + block_naive.x - 1) / block_naive.x,
                    (m + block_naive.y - 1) / block_naive.y);
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, (size_t)m * n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, (size_t)n * p * sizeof(float), cudaMemcpyHostToDevice));
    naive_cuda_matmul<<<grid_naive, block_naive>>>(d_C, d_A, d_B, m, n, p);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_C_naive, d_C, (size_t)m * p * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float time_naive_ms;
    CUDA_CHECK(cudaEventElapsedTime(&time_naive_ms, start, stop));
    float naive_time = time_naive_ms / 1000.0f;
    write_matrix(result_naive, h_C_naive, m, p);
    std::cerr << "result naive: " << h_C_naive << std::endl;
    bool correct_naive = validate_result(result_naive, reference);
    std::cout << "Naive CUDA " << (correct_naive ? "OK" : "FAIL")
              << ", time = " << naive_time << " s\n";

    // ----- Tiled CUDA -----
    uint32_t tile_width = 16;
    dim3 block_tiled(tile_width, tile_width);
    dim3 grid_tiled((p + tile_width - 1) / tile_width,
                    (m + tile_width - 1) / tile_width);
    size_t shared_mem_size = 2 * tile_width * tile_width * sizeof(float);
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, (size_t)m * n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, (size_t)n * p * sizeof(float), cudaMemcpyHostToDevice));
    tiled_cuda_matmul<<<grid_tiled, block_tiled, shared_mem_size>>>(
        d_C, d_A, d_B, m, n, p, tile_width);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_C_tiled, d_C, (size_t)m * p * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float time_tiled_ms;
    CUDA_CHECK(cudaEventElapsedTime(&time_tiled_ms, start, stop));
    float tiled_time = time_tiled_ms / 1000.0f;
    write_matrix(result_tiled, h_C_tiled, m, p);
    std::cerr << "result tiled: " << h_C_tiled << std::endl;
    bool correct_tiled = validate_result(result_tiled, reference);
    std::cout << "Tiled CUDA " << (correct_tiled ? "OK" : "FAIL")
              << ", time = " << tiled_time << " s\n";

    // Compute and print speedup of tiled over naive
    float speedup_naive = naive_time / tiled_time;
    std::cout << "Speedup (Tiled vs Naive): " 
            << speedup_naive << "×\n";

    // Clean up
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    std::free(h_A);
    std::free(h_B);
    std::free(h_C_naive);
    std::free(h_C_tiled);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return EXIT_SUCCESS;
}
