#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <iomanip>

struct Result {
	int case_number;
	int m, n, p;
	double naive_cuda_time;
	double tiled_cuda_time;
};

int main() {
	std::vector<Result> results;
	
	// Process all test cases
	for (int case_number = 0; case_number <= 9; ++case_number) {
		Result r;
		r.case_number = case_number;
		
		// Read matrix dimensions
		std::string data_dir = "data/" + std::to_string(case_number) + "/";
		std::ifstream input0_file(data_dir + "input0.raw", std::ios::binary);
		std::ifstream input1_file(data_dir + "input1.raw", std::ios::binary);
		
		if (!input0_file || !input1_file) {
			std::cerr << "Failed to open input files for case " << case_number << std::endl;
			continue;
		}
		
		input0_file.read(reinterpret_cast<char*>(&r.m), sizeof(uint32_t));
		input0_file.read(reinterpret_cast<char*>(&r.n), sizeof(uint32_t));
		
		uint32_t n2, p;
		input1_file.read(reinterpret_cast<char*>(&n2), sizeof(uint32_t));
		input1_file.read(reinterpret_cast<char*>(&r.p), sizeof(uint32_t));
		
		// Read timing results
		std::string cuda_results_file = data_dir + "cuda_results.txt";
		std::ifstream cuda_file(cuda_results_file);
		if (cuda_file) {
			cuda_file >> r.naive_cuda_time >> r.tiled_cuda_time;
		} else {
			r.naive_cuda_time = 0.0;
			r.tiled_cuda_time = 0.0;
		}
		
		results.push_back(r);
	}
	
	// Generate the table
	std::cout << "| Test Case | Dimensions (m×n×p) | Naive CUDA (s) | Tiled CUDA (s) | Speedup |" << std::endl;
	std::cout << "|-----------|-------------------|----------------|----------------|---------|" << std::endl;
	
	for (const auto& r : results) {
		double speedup = r.naive_cuda_time / r.tiled_cuda_time;
		
		std::cout << "| " << std::setw(9) << r.case_number << " | "
				  << std::setw(17) << r.m << "×" << r.n << "×" << r.p << " | "
				  << std::fixed << std::setprecision(6)
				  << std::setw(14) << r.naive_cuda_time << " | "
				  << std::setw(14) << r.tiled_cuda_time << " | "
				  << std::setw(7) << speedup << " |" << std::endl;
	}
	
	return 0;
}
