#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <omp.h>
#include <cmath>
#include <windows.h>
#include <algorithm>
#include <iomanip>
#include <string.h>
#include <cuda_runtime.h>
#include <locale>
#include <codecvt>
#include <io.h>
#include <fcntl.h>

std::wstring getParentPath();
std::wstring s2ws(const std::string &str);

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

    uint32_t row = blockIdx.y * tile_width + threadIdx.y;
    uint32_t col = blockIdx.x * tile_width + threadIdx.x;

    float sum = 0.0f;


    extern __shared__ float shared_mem[];
    float *As = shared_mem;
    float *Bs = shared_mem + tile_width * tile_width;


    for (uint32_t t = 0; t < (n + tile_width - 1) / tile_width; ++t) {

        uint32_t tiledRow = row;
        uint32_t tiledCol = t * tile_width + threadIdx.x;
        if (tiledRow < m && tiledCol < n)
            As[threadIdx.y * tile_width + threadIdx.x] = A[tiledRow * n + tiledCol];
        else
            As[threadIdx.y * tile_width + threadIdx.x] = 0.0f;


        tiledRow = t * tile_width + threadIdx.y;
        tiledCol = col;
        if (tiledRow < n && tiledCol < p)
            Bs[threadIdx.y * tile_width + threadIdx.x] = B[tiledRow * p + tiledCol];
        else
            Bs[threadIdx.y * tile_width + threadIdx.x] = 0.0f;

        __syncthreads();

        for (uint32_t k = 0; k < tile_width; ++k) {
            sum += As[threadIdx.y * tile_width + k] * Bs[k * tile_width + threadIdx.x];
        }

        __syncthreads();
    }


    if (row < m && col < p) {
        C[row * p + col] = sum;
    }
}

bool validate_result(const std::wstring &path) {
    std::wstring result_file = path + L"\\result.raw";
    std::wstring output_file = path + L"\\output.raw";

    std::wifstream result(result_file);
    std::wifstream output(output_file);

    if (!result.is_open() || !output.is_open()) {
        std::wcerr << L"Error opening result or output file in path: " << path << std::endl;
        return false;
    }

    std::wstring result_line, output_line;
    int line_number = 1;

    while (std::getline(result, result_line) && std::getline(output, output_line)) {
        std::wistringstream result_stream(result_line);
        std::wistringstream output_stream(output_line);

        float result_value, output_value;
        int value_index = 1;

        while (result_stream >> result_value && output_stream >> output_value) {

            if (std::abs(result_value - output_value) > 1e-5) {
                std::cerr << "Mismatch found at line " << line_number << ", value " << value_index << ":\n"
                    << "Result: " << result_value << ", Output: " << output_value << std::endl;
                result.close();
                output.close();
                return false;
            }
            value_index++;
        }


        if ((result_stream >> result_value) || (output_stream >> output_value)) {
            std::cerr << "Mismatch in number of values at line " << line_number << "." << std::endl;
            result.close();
            output.close();
            return false;
        }

        line_number++;
    }

    return true;
}

int main(int argc, char *argv[]) {

	double start_time, elapsed_time, naive_cuda_time, tiled_cuda_time;

    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <case_number>" << std::endl;
        return 1;
    }

    int case_number = std::atoi(argv[1]);
    if (case_number < 0 || case_number > 9) {
        std::cerr << "Case number must be between 0 and 9" << std::endl;
        return 1;
    }

    std::wstring parentPath = getParentPath();
    std::wcout << L"Parent Path: " << parentPath << std::endl;

    std::wstring root_path = parentPath + L"\\Homework-3\\data\\";
    std::wcout << L"Old Root Path: " << root_path << std::endl;


    std::wstring path = parentPath;
    for (int i = 0; i < 3; ++i) {
        size_t pos = path.find_last_of(L"\\/");
        if (pos == std::string::npos) {
            path.clear();
            break;
        }
        path = path.substr(0, pos);
    }
    root_path = path + L"\\Homework-3\\data\\";
    std::wcout << L"New Root Path: " << root_path << std::endl;


    std::wstring folder = std::to_wstring(case_number) + L"\\";
    std::wstring input0_file = root_path + folder + L"input0.raw";
    std::wstring input1_file = root_path + folder + L"input1.raw";
    std::wstring result_file = root_path + folder + L"result.raw";
    std::wstring reference_file = root_path + folder + L"output.raw";

    std::ifstream ifs;


    std::wstring wroot_path = root_path;
    std::wstring wfolder = folder;
    std::wstring winput0 = wroot_path + wfolder + L"input0.raw";
    std::wstring winput1 = wroot_path + wfolder + L"input1.raw";

    std::wstring result_dir = root_path + folder + L"result.raw";

    std::wcout << L"Trying to open: " << winput0 << std::endl;
    std::wcout << L"Trying to open: " << winput1 << std::endl;

    if (_waccess(winput0.c_str(), 4) != 0) {
        std::wcerr << L"File does not exist or is not readable: " << winput0 << std::endl;
    }
    if (_waccess(winput1.c_str(), 4) != 0) {
        std::wcerr << L"File does not exist or is not readable: " << winput1 << std::endl;
    }


    std::wifstream input0(winput0);
    std::wifstream input1(winput1);

    if (!input0.is_open() || !input1.is_open()) {

        std::cerr << "\nError opening input files." << std::endl;



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



    if (!input1.is_open()) {
        std::cerr << "Error opening input1.raw" << std::endl;
        free(A);
        return -1;
    }
    float *B = new float[n * p];

    for (int i = 0; i < n * p; ++i) input1 >> B[i];
    input1.close();

    
	// Naive CUDA Matmul
    float naive_cuda_time_ms = 0.0f;
    cudaEvent_t start_naive, stop_naive;
    cudaEventCreate(&start_naive);
    cudaEventCreate(&stop_naive);

    cudaEventRecord(start_naive, 0);
    float *C_naive = new float[m * p];


    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, m * n * sizeof(float));
    cudaMalloc(&d_B, n * p * sizeof(float));
    cudaMalloc(&d_C, m * p * sizeof(float));


    cudaMemcpy(d_A, A, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, n * p * sizeof(float), cudaMemcpyHostToDevice);


    dim3 blockDim(16, 16);
    dim3 gridDim((p + blockDim.x - 1) / blockDim.x, (m + blockDim.y - 1) / blockDim.y);


    naive_cuda_matmul<<<gridDim, blockDim>>>(d_C, d_A, d_B, m, n, p);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << std::endl;
    }
    cudaDeviceSynchronize();


    cudaMemcpy(C_naive, d_C, m * p * sizeof(float), cudaMemcpyDeviceToHost);



    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventRecord(stop_naive, 0);
    cudaEventSynchronize(stop_naive);

    cudaEventElapsedTime(&naive_cuda_time_ms, start_naive, stop_naive);

    bool naive_correct = validate_result(root_path + folder);
    if (!naive_correct) {
        std::cerr << "Naive CUDA result validation failed." << std::endl;
        free(A);
        delete[] B;
        delete[] C_naive;
        return -1;
    }



	// Tiled CUDA Matmul
    float tiled_cuda_time_ms = 0.0f;
    cudaEvent_t start_tiled, stop_tiled;
    cudaEventCreate(&start_tiled);
    cudaEventCreate(&stop_tiled);

    cudaEventRecord(start_tiled, 0);
    float *C_tiled = new float[m * p];


    cudaMalloc(&d_C, m * p * sizeof(float));


    uint32_t tile_width = 16;
    dim3 blockDimTiled(tile_width, tile_width);
    dim3 gridDimTiled((p + tile_width - 1) / tile_width, (m + tile_width - 1) / tile_width);
    size_t shared_mem_size = 2 * tile_width * tile_width * sizeof(float);


    tiled_cuda_matmul<<<gridDimTiled, blockDimTiled, shared_mem_size>>>(d_C, d_A, d_B, m, n, p, tile_width);
    cudaDeviceSynchronize();


    cudaMemcpy(C_tiled, d_C, m * p * sizeof(float), cudaMemcpyDeviceToHost);


    cudaFree(d_C);

    cudaEventRecord(stop_tiled, 0);
    cudaEventSynchronize(stop_tiled);

    cudaEventElapsedTime(&tiled_cuda_time_ms, start_tiled, stop_tiled);


	bool tiled_correct = validate_result(root_path + folder);
	if (!tiled_correct) {
		std::cerr << "Tiled CUDA result validation failed." << std::endl;
		free(A);
		delete[] B;
		delete[] C_naive;
		delete[] C_tiled;
		return -1;
	}




    elapsed_time = omp_get_wtime() - start_time;
    tiled_cuda_time = elapsed_time;


    std::cout << "Case " << case_number << " (" << m << "x" << n << "x" << p << "):\n";
	std::cout << "Naive CUDA time: " << naive_cuda_time_ms / 1000.0f << " seconds\n";    
    std::cout << "Tiled CUDA time: " << tiled_cuda_time_ms / 1000.0f << " seconds\n";


    free(A);
    delete[] B;
    delete[] C_naive;
    delete[] C_tiled;


    cudaEventDestroy(start_tiled);
    cudaEventDestroy(stop_tiled);

    return 0;
}



std::wstring getParentPath() {
    wchar_t buffer[MAX_PATH];
    GetModuleFileNameW(NULL, buffer, MAX_PATH);
    std::wstring executablePath(buffer);


    size_t pos = executablePath.find_last_of(L"\\/");
    for (int i = 0; i < 2 && pos != std::wstring::npos; ++i) {
        pos = executablePath.find_last_of(L"\\/", pos - 1);
    }
    if (pos == std::wstring::npos) {
        return L"";
    }
    return executablePath.substr(0, pos);
}


std::wstring s2ws(const std::string &str) {
    int size_needed = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), (int)str.size(), NULL, 0);
    std::wstring wstrTo(size_needed, 0);
    MultiByteToWideChar(CP_UTF8, 0, str.c_str(), (int)str.size(), &wstrTo[0], size_needed);
    return wstrTo;
}