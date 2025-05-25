#include <iostream>
#include <fstream>
#include <string>
#include <cstdint>
#include <cmath>
#include <iomanip>
#include <omp.h>
#include <algorithm>

// Naive matrix multiplication
void naive_matmul(float* C, float* A, float* B, uint32_t m, uint32_t n, uint32_t p) {
    for (uint32_t i = 0; i < m; ++i) {
        for (uint32_t j = 0; j < p; ++j) {
            float sum = 0.0f;
            for (uint32_t k = 0; k < n; ++k) {
                sum += A[i * n + k] * B[k * p + j];
            }
            C[i * p + j] = sum;
        }
    }
}

// Blocked matrix multiplication
void blocked_matmul(float* C, float* A, float* B, uint32_t m, uint32_t n, uint32_t p, uint32_t block_size) {
    // Initialize output matrix to zero
    std::fill(C, C + m * p, 0.0f);

    for (uint32_t ii = 0; ii < m; ii += block_size) {
        for (uint32_t jj = 0; jj < p; jj += block_size) {
            for (uint32_t kk = 0; kk < n; kk += block_size) {
                uint32_t i_end = std::min(ii + block_size, m);
                uint32_t j_end = std::min(jj + block_size, p);
                uint32_t k_end = std::min(kk + block_size, n);

                for (uint32_t i = ii; i < i_end; ++i) {
                    for (uint32_t j = jj; j < j_end; ++j) {
                        float sum = 0.0f;
                        for (uint32_t k = kk; k < k_end; ++k) {
                            sum += A[i * n + k] * B[k * p + j];
                        }
                        C[i * p + j] += sum;
                    }
                }
            }
        }
    }
}

// Parallel matrix multiplication using OpenMP
void parallel_matmul(float* C, float* A, float* B, uint32_t m, uint32_t n, uint32_t p) {
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(m); ++i) {
        for (uint32_t j = 0; j < p; ++j) {
            float sum = 0.0f;
            for (uint32_t k = 0; k < n; ++k) {
                sum += A[i * n + k] * B[k * p + j];
            }
            C[i * p + j] = sum;
        }
    }
}

bool load_matrix(const std::string& path, float*& data, uint32_t& rows, uint32_t& cols) {
    std::ifstream file(path);
    if (!file) {
        std::cerr << "Failed to open " << path << std::endl;
        return false;
    }
    file >> rows >> cols;
    data = new float[rows * cols];
    for (uint32_t i = 0; i < rows * cols; ++i) {
        file >> data[i];
    }
    return true;
}

bool write_matrix(const std::string& path, float* data, uint32_t rows, uint32_t cols) {
    std::ofstream file(path);
    if (!file) {
        std::cerr << "Failed to write to " << path << std::endl;
        return false;
    }
    file << rows << " " << cols << "\n";
    file << std::fixed << std::setprecision(2);

    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < cols; ++j) {
            file << data[i * cols + j] << " ";
        }
        file << "\n";
    }
    return true;
}

bool validate_result(const std::string& expected_path, const std::string& result_path) {
    std::ifstream expected(expected_path), result(result_path);
    if (!expected || !result) {
        std::cerr << "Validation failed: could not open output files.\n";
        return false;
    }

    uint32_t e_rows, e_cols, r_rows, r_cols;
    expected >> e_rows >> e_cols;
    result >> r_rows >> r_cols;

    if (e_rows != r_rows || e_cols != r_cols) {
        std::cerr << "Dimension mismatch in validation.\n";
        return false;
    }

    for (uint32_t i = 0; i < e_rows * e_cols; ++i) {
        float a, b;
        expected >> a;
        result >> b;
        if (std::abs(a - b) > 1e-3) {
            std::cerr << "Mismatch at index " << i << ": " << a << " vs " << b << "\n";
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {

    omp_set_num_threads(4);

    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <test_case_number (0-9)>" << std::endl;
        return 1;
    }

    std::string case_num = argv[1];
    std::string folder = "data/" + case_num + "/";
    std::string inputA = folder + "input0.raw";
    std::string inputB = folder + "input1.raw";
    std::string outputPath = folder + "result.raw";
    std::string expectedPath = folder + "output.raw";

    float* A = nullptr;
    float* B = nullptr;
    uint32_t m, n, nB, p;

    if (!load_matrix(inputA, A, m, n)) return 1;
    if (!load_matrix(inputB, B, nB, p)) return 1;

    if (n != nB) {
        std::cerr << "Matrix dimensions incompatible for multiplication.\n";
        delete[] A;
        delete[] B;
        return 1;
    }

    // Allocate result matrices
    float* C_naive = new float[m * p]();
    float* C_blocked = new float[m * p]();
    float* C_parallel = new float[m * p]();

    // Test naive multiplication
    // double start_time = omp_get_wtime();
    // naive_matmul(C_naive, A, B, m, n, p);
    // double naive_time = omp_get_wtime() - start_time;
    // write_matrix(outputPath, C_naive, m, p);
    // bool naive_correct = validate_result(expectedPath, outputPath);

    // Modified timing measurement in main()
    const int iterations = 5;  // Run multiple times for more accurate timing

    // Test naive multiplication
    double naive_time = 0;
    for (int i = 0; i < iterations; ++i) {
        std::fill(C_naive, C_naive + m * p, 0.0f);
        double start = omp_get_wtime();
        naive_matmul(C_naive, A, B, m, n, p);
        naive_time += omp_get_wtime() - start;
    }
    naive_time /= iterations;
    bool naive_correct = validate_result(expectedPath, outputPath);

    // Test blocked multiplication
    double blocked_time = 0;
    for (int i = 0; i < iterations; ++i) {
        std::fill(C_blocked, C_blocked + m * p, 0.0f);
        double start = omp_get_wtime();
        //smaller block size for small matrices
        uint32_t block_size = (m <= 128 && n <= 128 && p <= 128) ? 2 : 8;
        blocked_matmul(C_blocked, A, B, m, n, p, block_size);
        //blocked_matmul(C_blocked, A, B, m, n, p, 32);
        blocked_time += omp_get_wtime() - start;
    }
    blocked_time /= iterations;

    // Test parallel multiplication
    double parallel_time = 0;
    for (int i = 0; i < iterations; ++i) {
        std::fill(C_parallel, C_parallel + m * p, 0.0f);
        double start = omp_get_wtime();
        parallel_matmul(C_parallel, A, B, m, n, p);
        parallel_time += omp_get_wtime() - start;
    }
    parallel_time /= iterations;

    // // Test blocked multiplication (using block size 32)
    // start_time = omp_get_wtime();
    // blocked_matmul(C_blocked, A, B, m, n, p, 4);
    // double blocked_time = omp_get_wtime() - start_time;
    // write_matrix(outputPath, C_blocked, m, p);
    bool blocked_correct = validate_result(expectedPath, outputPath);
    //
    // // Test parallel multiplication
    // start_time = omp_get_wtime();
    // parallel_matmul(C_parallel, A, B, m, n, p);
    // double parallel_time = omp_get_wtime() - start_time;
    // write_matrix(outputPath, C_parallel, m, p);
    bool parallel_correct = validate_result(expectedPath, outputPath);

    // Print results
    std::cout << "Case " << case_num << " (" << m << "x" << n << "x" << p << "):\n";
    std::cout << "Naive time: " << naive_time << " seconds - "
              << (naive_correct ? "CORRECT" : "INCORRECT") << "\n";
    std::cout << "Blocked time: " << blocked_time << " seconds - "
              << (blocked_correct ? "CORRECT" : "INCORRECT") << "\n";
    std::cout << "Parallel time: " << parallel_time << " seconds - "
              << (parallel_correct ? "CORRECT" : "INCORRECT") << "\n";

    // if (naive_correct) {
    //     std::cout << "Blocked speedup: " << (naive_time / blocked_time) << "x\n";
    //     std::cout << "Parallel speedup: " << (naive_time / parallel_time) << "x\n";
    // }

    // Modified speedup calculation
    if (naive_correct) {
        if (blocked_time > 0) {
            std::cout << "Blocked speedup: " << (naive_time / blocked_time) << "x\n";
        } else {
            std::cout << "Blocked speedup: Too fast to measure\n";
        }
        if (parallel_time > 0) {
            std::cout << "Parallel speedup: " << (naive_time / parallel_time) << "x\n";
        } else {
            std::cout << "Parallel speedup: Too fast to measure\n";
        }
    }

    // Clean up
    delete[] A;
    delete[] B;
    delete[] C_naive;
    delete[] C_blocked;
    delete[] C_parallel;

    return 0;
}