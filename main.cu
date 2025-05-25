#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <vector>
#include <cstring>
#include <cuda_runtime.h>

// Error checking macro for CUDA API calls
#define CHECK_CUDA_ERROR(err) do { \
	if (err != cudaSuccess) { \
		fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
		exit(EXIT_FAILURE); \
	} \
} while(0)

// Each thread computes one element of C
// Global memory access pattern is inefficient but straightforward
__global__ void naive_cuda_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p) {
	uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
	uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (row < m && col < p) {
		float sum = 0.0f;
		for (int k = 0; k < n; ++k) {
			sum += A[row * n + k] * B[k * p + col];
		}
		C[row * p + col] = sum;
	}
}

// Uses shared memory to reduce global memory access
// Each block loads tiles of matrices into fast shared memory
__global__ void tiled_cuda_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p, uint32_t tile_width) {
	extern __shared__ float shared_mem[];
	float *tile_A = shared_mem;
	float *tile_B = shared_mem + tile_width * tile_width;
	
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	
	int row = by * tile_width + ty;
	int col = bx * tile_width + tx;
	
	float sum = 0.0f;
	
	// Process matrix multiplication tile by tile
	for (int t = 0; t < (n + tile_width - 1) / tile_width; ++t) {
		// Load tile from matrix A into shared memory
		if (row < m && t * tile_width + tx < n) {
			tile_A[ty * tile_width + tx] = A[row * n + (t * tile_width + tx)];
		} else {
			tile_A[ty * tile_width + tx] = 0.0f;
		}
		
		// Load tile from matrix B into shared memory
		if (t * tile_width + ty < n && col < p) {
			tile_B[ty * tile_width + tx] = B[(t * tile_width + ty) * p + col];
		} else {
			tile_B[ty * tile_width + tx] = 0.0f;
		}
		
		__syncthreads();
		
		// Compute partial dot product using data in shared memory
		if (row < m && col < p) {
			for (int k = 0; k < tile_width; ++k) {
				sum += tile_A[ty * tile_width + k] * tile_B[k * tile_width + tx];
			}
		}
		
		__syncthreads();
	}
	
	if (row < m && col < p) {
		C[row * p + col] = sum;
	}
}

// Validates results against reference with small tolerance for floating-point differences
bool validate_result(const std::string &result_file, const std::string &reference_file) {
	std::ifstream result_stream(result_file, std::ios::binary);
	std::ifstream reference_stream(reference_file, std::ios::binary);
	
	if (!result_stream || !reference_stream) {
		std::cerr << "Failed to open result or reference file" << std::endl;
		return false;
	}
	
	result_stream.seekg(0, std::ios::end);
	reference_stream.seekg(0, std::ios::end);
	
	std::streamsize result_size = result_stream.tellg();
	std::streamsize reference_size = reference_stream.tellg();
	
	if (result_size != reference_size) {
		std::cerr << "Result and reference file sizes don't match" << std::endl;
		return false;
	}
	
	result_stream.seekg(0, std::ios::beg);
	reference_stream.seekg(0, std::ios::beg);
	
	const int BUFFER_SIZE = 4096;
	char result_buffer[BUFFER_SIZE], reference_buffer[BUFFER_SIZE];
	bool is_valid = true;
	float tolerance = 1e-3f;
	
	while (result_stream && reference_stream) {
		result_stream.read(result_buffer, BUFFER_SIZE);
		reference_stream.read(reference_buffer, BUFFER_SIZE);
		
		size_t bytes_read = result_stream.gcount();
		if (bytes_read == 0) break;
		
		for (size_t i = 0; i < bytes_read; i += sizeof(float)) {
			float result_val = *reinterpret_cast<float*>(&result_buffer[i]);
			float ref_val = *reinterpret_cast<float*>(&reference_buffer[i]);
			if (std::abs(result_val - ref_val) > tolerance) {
				is_valid = false;
				break;
			}
		}
		
		if (!is_valid) break;
	}
	
	return is_valid;
}

// Read matrix dimension and data from binary file format
std::vector<float> read_matrix_from_file(const std::string &filename, uint32_t &rows, uint32_t &cols) {
	std::ifstream file(filename, std::ios::binary);
	if (!file) {
		std::cerr << "Failed to open file: " << filename << std::endl;
		exit(EXIT_FAILURE);
	}
	
	file.read(reinterpret_cast<char*>(&rows), sizeof(uint32_t));
	file.read(reinterpret_cast<char*>(&cols), sizeof(uint32_t));
	
	std::vector<float> matrix(rows * cols);
	file.read(reinterpret_cast<char*>(matrix.data()), rows * cols * sizeof(float));
	
	return matrix;
}

// Write matrix dimensions and data to binary file format
void write_matrix_to_file(const std::string &filename, const float *matrix, uint32_t rows, uint32_t cols) {
	std::ofstream file(filename, std::ios::binary);
	if (!file) {
		std::cerr << "Failed to open file for writing: " << filename << std::endl;
		exit(EXIT_FAILURE);
	}
	
	file.write(reinterpret_cast<const char*>(&rows), sizeof(uint32_t));
	file.write(reinterpret_cast<const char*>(&cols), sizeof(uint32_t));
	file.write(reinterpret_cast<const char*>(matrix), rows * cols * sizeof(float));
}

// Measure execution time of matrix multiplication, including memory transfers
double measure_cuda_matmul(int case_number, uint32_t m, uint32_t n, uint32_t p, 
						  const std::vector<float>& A, const std::vector<float>& B, 
						  std::vector<float>& C, bool use_tiled, uint32_t tile_width = 16) {
	float *d_A, *d_B, *d_C;
	
	// Allocate device memory for matrices
	CHECK_CUDA_ERROR(cudaMalloc(&d_A, m * n * sizeof(float)));
	CHECK_CUDA_ERROR(cudaMalloc(&d_B, n * p * sizeof(float)));
	CHECK_CUDA_ERROR(cudaMalloc(&d_C, m * p * sizeof(float)));
	
	// Setup timing using CUDA events for higher precision
	cudaEvent_t start, stop;
	CHECK_CUDA_ERROR(cudaEventCreate(&start));
	CHECK_CUDA_ERROR(cudaEventCreate(&stop));
	
	CHECK_CUDA_ERROR(cudaEvent_RECORD(start));
	
	// Copy input matrices from host to device
	CHECK_CUDA_ERROR(cudaMemcpy(d_A, A.data(), m * n * sizeof(float), cudaMemcpyHostToDevice));
	CHECK_CUDA_ERROR(cudaMemcpy(d_B, B.data(), n * p * sizeof(float), cudaMemcpyHostToDevice));
	
	// Configure grid and block dimensions
	dim3 threadsPerBlock(tile_width, tile_width);
	dim3 blocksPerGrid((p + threadsPerBlock.x - 1) / threadsPerBlock.x,
					  (m + threadsPerBlock.y - 1) / threadsPerBlock.y);
	
	// Launch appropriate kernel based on algorithm choice
	if (use_tiled) {
		size_t shared_mem_size = 2 * tile_width * tile_width * sizeof(float);
		tiled_cuda_matmul<<<blocksPerGrid, threadsPerBlock, shared_mem_size>>>(
			d_C, d_A, d_B, m, n, p, tile_width);
	} else {
		naive_cuda_matmul<<<blocksPerGrid, threadsPerBlock>>>(d_C, d_A, d_B, m, n, p);
	}
	
	// Check for kernel errors
	CHECK_CUDA_ERROR(cudaGetLastError());
	CHECK_CUDA_ERROR(cudaDeviceSynchronize());
	
	// Copy result from device to host
	C.resize(m * p);
	CHECK_CUDA_ERROR(cudaMemcpy(C.data(), d_C, m * p * sizeof(float), cudaMemcpyDeviceToHost));
	
	CHECK_CUDA_ERROR(cudaEventRecord(stop));
	CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
	
	// Calculate elapsed time in milliseconds
	float milliseconds = 0;
	CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
	
	// Free device memory
	CHECK_CUDA_ERROR(cudaFree(d_A));
	CHECK_CUDA_ERROR(cudaFree(d_B));
	CHECK_CUDA_ERROR(cudaFree(d_C));
	
	CHECK_CUDA_ERROR(cudaEventDestroy(start));
	CHECK_CUDA_ERROR(cudaEventDestroy(stop));
	
	return milliseconds / 1000.0; // Convert to seconds
}

int main(int argc, char *argv[]) {
	if (argc != 2) {
		std::cerr << "Usage: " << argv[0] << " <case_number>" << std::endl;
		return 1;
	}
	
	int case_number = std::stoi(argv[1]);
	if (case_number < 0 || case_number > 9) {
		std::cerr << "Case number must be between 0 and 9" << std::endl;
		return 1;
	}
	
	// Set up file paths for the current test case
	std::string data_dir = "data/" + std::to_string(case_number) + "/";
	std::string input0_file = data_dir + "input0.raw";
	std::string input1_file = data_dir + "input1.raw";
	std::string output_file = data_dir + "output.raw";
	std::string naive_result_file = data_dir + "naive_result.raw";
	std::string tiled_result_file = data_dir + "tiled_result.raw";
	
	// Read input matrices A and B
	uint32_t m, n, p, n2;
	std::vector<float> A = read_matrix_from_file(input0_file, m, n);
	std::vector<float> B = read_matrix_from_file(input1_file, n2, p);
	
	if (n != n2) {
		std::cerr << "Matrix dimensions don't match for multiplication" << std::endl;
		return 1;
	}
	
	std::vector<float> C_naive, C_tiled;
	
	// Run naive CUDA matrix multiplication
	double naive_cuda_time = measure_cuda_matmul(case_number, m, n, p, A, B, C_naive, false);
	
	// Validate naive result against reference
	write_matrix_to_file(naive_result_file, C_naive.data(), m, p);
	bool naive_valid = validate_result(naive_result_file, output_file);
	
	// Run tiled CUDA matrix multiplication (tile_width = 16 gives good balance)
	uint32_t tile_width = 16;
	double tiled_cuda_time = measure_cuda_matmul(case_number, m, n, p, A, B, C_tiled, true, tile_width);
	
	// Validate tiled result against reference
	write_matrix_to_file(tiled_result_file, C_tiled.data(), m, p);
	bool tiled_valid = validate_result(tiled_result_file, output_file);
	
	// Print performance results in the format expected by run_tests.sh
	std::cout << "Case " << case_number << " (" << m << "x" << n << "x" << p << "):\n";
	std::cout << "Naive CUDA time: " << naive_cuda_time << " seconds";
	std::cout << (naive_valid ? " [VALID]" : " [INVALID]") << std::endl;
	std::cout << "Tiled CUDA time: " << tiled_cuda_time << " seconds";
	std::cout << (tiled_valid ? " [VALID]" : " [INVALID]") << std::endl;
	std::cout << "Tiled CUDA speedup over naive: " << naive_cuda_time / tiled_cuda_time << "x" << std::endl;
	
	CHECK_CUDA_ERROR(cudaDeviceReset());
	
	return 0;
}