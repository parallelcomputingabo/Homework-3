#include "device_launch_parameters.h"
#include <iostream>
#include <fstream>
#include <string>
#include <omp.h>
#include <cmath>
#include <iomanip>
#include <cuda_runtime.h>
#include <cstdio>
#include <filesystem>
#include <stdio.h>

using namespace std;

struct Matrix {
    int x, y;
    double* dataRowMajorOrder;
};

Matrix* getMatrix(string filePath) {
    // Read file
    ifstream file(filePath);

    if (!file) {
        Matrix* result = new Matrix;
        return result;
    }

    // Max size of a line (size can be anything if available in memory. 100000 in this case
    char line[100000];

    // Dimensions
    int x = 0;
    int y = 0;

    // iteration/list element
    int i = 0;

    // Matrix data, placed somewhere in memory
    double* collection = nullptr;

    while (file.getline(line, 100000)) {
        // Loop through the file, line by line.
        // A line is a character list consisting of 100000 slots

        // Split line by blank space into token
        char* token = strtok(line, " ");
        while (token != nullptr) {
            // Iterate over one token at a time

            // First line in each input file represents dimensions X Y 
            if (x == 0) {
                // If X has not been set, then pick first token from first line,
                // Assuming this is the first line and this is the first token
                x = atoi(token);
            }
            else if (y == 0) {
                // If Y has not been set, then pick second token from first line,
                // Assuming this is the first line and this is the second token iteration
                y = atoi(token);

                // Now, set the total slots (x * y) in memory 
                collection = new double[x * y];
            }
            else {
                // Insert token into collection
                // Assuming this is the second line and onwards
                collection[i++] = atof(token);
            }

            token = strtok(nullptr, " ");
        }
    }

    file.close();

    // Define the Matrix
    Matrix* result = new Matrix;

    // Matrix dimensions
    result->x = x;
    result->y = y;

    // Data collection of tokens from second line and onwards
    result->dataRowMajorOrder = collection;

    return result;
}

__global__ void naive_cuda_matmul(double* C, double* A, double* B, uint32_t m, uint32_t n, uint32_t p) {
    // A = First matrix, in the form of one-dimensional/row-major list
    // B = Second Matrix, in the form of one-dimensional/row-major list
    // C = A x B, in the form of one-dimensional/row-major list
    //
    // m = number of rows in matrix A
    // n = number of columns in matrix A
    // p = number of columns in matrix B

    // TODO: Implement naive matrix multiplication C = A x B
    // A is m x n, B is n x p, C is m x p

    // In assignment2, a multidimensional loop was used, which
    // could be paralellized thanks to openMP. CUDA has no such
    // feature, thus the rewriting of the multiplication algorithm
    // with respect to how grid, blocks and threads works.

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < m && j < p) {
        double sum = 0.0;
        for (int k = 0; k < n; ++k) {
            sum += A[i * n + k] * B[k * p + j];
        }
        C[i * p + j] = sum;
    }
}

__global__ void tiled_cuda_matmul(double* C, double* A, double* B, uint32_t m, uint32_t n, uint32_t p, uint32_t tile_width) {

}

bool compareFiles(string originPath, string targetPath) {
    ifstream originalFile(originPath);
    ifstream targetFile(targetPath);

    if (!originalFile || !targetFile) {
        return false;
    }

    string originalFileContents;
    string targetFileContents;

    while (getline(originalFile, originalFileContents) && getline(targetFile, targetFileContents)) {
        string originalFileLetters = "";
        string targetFileLetters = "";

        for (char character : targetFileContents) {
            if (character != ' ') {
                targetFileLetters += character;
            }
        }

        for (char character : originalFileContents) {
            if (character != ' ') {
                originalFileLetters += character;
            }
        }

        if (targetFileLetters != originalFileLetters) {
            targetFile.close();
            originalFile.close();
            return false;
        }
    }

    targetFile.close();
    originalFile.close();
    return true;
}

double measure_performance(double* CPU_C, double* CPU_A, double* CPU_B, uint32_t m, uint32_t n, uint32_t p) {
    dim3 blockSize(32, 32);
	dim3 gridSize((p + 31) / 32, (m + 31) / 32);

    double* GPU_A;
    double* GPU_B;
    double* GPU_C;

    size_t sizeA = sizeof(double) * m * n;
    size_t sizeB = sizeof(double) * n * p;
    size_t sizeC = sizeof(double) * m * p;
	

    // Allocate GPU memory
    cudaMalloc(&GPU_A, sizeA);
    cudaMalloc(&GPU_B, sizeB);
    cudaMalloc(&GPU_C, sizeC);

    // Copy matrices from CPU to GPU
    cudaMemcpy(GPU_A, CPU_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(GPU_B, CPU_B, sizeB, cudaMemcpyHostToDevice);

	
    cudaEvent_t start;
	cudaEvent_t stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // Start recording the performance
    cudaEventRecord(start);
    naive_cuda_matmul<<<gridSize, blockSize>>>(GPU_C, GPU_A, GPU_B, m, n, p);
	cudaDeviceSynchronize();

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        cout << "CUDA kernel launch failed: " << cudaGetErrorString(error) << endl;
    }
	
    cudaEventRecord(stop);
	cudaMemcpy(CPU_C, GPU_C, sizeC, cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);


    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

	
    // Delete memory and events from GPU
    cudaFree(GPU_A);
    cudaFree(GPU_B);
    cudaFree(GPU_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds;
}
int main(int argc, char* argv[]) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " "  << argv[1] << "<case_number>" << endl;
        return 1;
    }

    int case_number = atoi(argv[1]);
    if (case_number < 0 || case_number > 11) {
        cerr << "Case number must be between 0 and 11" << endl;
        return 1;
    }

    string folder = "./data/" + to_string(case_number) + "/";
    string input0_file = "./data/" + to_string(case_number) + "/input0.raw";
    string input1_file = "./data/" + to_string(case_number) + "/input1.raw";
    string naive_result_file = "./data/" + to_string(case_number) + "/naive_result.raw";
    string reference_file = "./data/" + to_string(case_number) + "/output.raw";

    Matrix* AMatrix = getMatrix(input0_file);
    Matrix* BMatrix = getMatrix(input1_file);

    int m = AMatrix->x;
    int n = AMatrix->y;
    int p = BMatrix->y;

    double *C_naive = new double[m * p];
		
    double naive_time = measure_performance(C_naive, AMatrix->dataRowMajorOrder, BMatrix->dataRowMajorOrder, m, n, p);

    ofstream naiveResultFile(naive_result_file);
    naiveResultFile << m << " " << p << "\n";
    int i = 0;
    int size = m * p;
    while (i < size) {
        bool isInteger = fabs(C_naive[i] - round(C_naive[i])) < 1e-6;

        if (i % p == p - 1 && i + 1 != size) {
            // If this is the last character in a line, then insert linebreaks
            // Some integers in the output file has a dot (e.g. 247.). This code makes sure to satisfy that in results.raw
            if (isInteger) {
                naiveResultFile << C_naive[i] << "." << "\n";
            }
            else {
                naiveResultFile << C_naive[i] << "\n";
            }
        }
        else {
            if (isInteger) {
                naiveResultFile << C_naive[i] << "." << " ";
            }
            else {
                naiveResultFile << C_naive[i] << " ";
            }
        }
        i++;
    }
    naiveResultFile.close();

    // Validate naive result
    bool naive_correct = compareFiles(reference_file, naive_result_file);
    if (!naive_correct) {
        std::cerr << "Naive result validation failed for case " << case_number << std::endl;
    }

    std::cout << "Case " << case_number << " (" << m << "x" << n << "x" << p << "):\n";
    std::cout << "Naive time: " << naive_time << " seconds\n";
	
	// Clean up
    delete[] AMatrix->dataRowMajorOrder;
    delete[] BMatrix->dataRowMajorOrder;
    delete[] C_naive;
    delete[] AMatrix;
    delete[] BMatrix;

    return 0;
}
