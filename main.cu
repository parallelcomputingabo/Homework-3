#include <iostream>
#include <fstream>
#include <string>
#include <cuda_runtime.h>
#include <time.h>
#include <sys/time.h>
#include <iomanip>

#define USECPSEC 1000000ULL
const int TILE_WIDTH=16;
using namespace std;
__global__ void naive_cuda_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p) {
    // TODO: Implement naive CUDA matrix multiplication
    int row=blockIdx.y*blockDim.y+threadIdx.y;
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    if(row<m && col<p){
    float sum=0;
    for(int i=0;i<n;i++)
        if(row*n+i<m*n && i*p+col<n*p)
            sum+=A[row*n+i]*B[i*p+col];
    C[row*p+col]=sum;
    }
}

__global__ void tiled_cuda_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p, uint32_t tile_width) {
    // TODO: Implement tiled CUDA matrix multiplication
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * tile_width + threadIdx.y;
    int col = blockIdx.x * tile_width + threadIdx.x;

    float value = 0;

    for (int t = 0; t < (n + tile_width - 1) / tile_width; ++t) {
        if (row < m && t * tile_width + threadIdx.x < n)
            tileA[threadIdx.y][threadIdx.x] = A[row * n + t * tile_width + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0;

        if (col < p && t * tile_width + threadIdx.y < n)
            tileB[threadIdx.y][threadIdx.x] = B[(t * tile_width + threadIdx.y) * p + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0;

        __syncthreads();

        for (int i = 0; i < tile_width; ++i)
            value += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];

        __syncthreads();
    }

    if (row < m && col < p)
        C[row * p + col] = value;
}

bool validate_result(const std::string &result_file, const std::string &reference_file) {
    // TODO: Implement result validation (same as Assignment 2)
    ifstream fptr_result(result_file);
    ifstream fptr_refer(reference_file);

    float temp1,temp2;
    while(!fptr_result.eof()) {
        fptr_result>>temp1;
        fptr_refer>>temp2;
        // I used deficit because of precision issue with floating numbers
        if(abs(temp1-temp2)>=0.5) {
            cout<<"Failed comparison: "<<temp1<<" "<<temp2<<endl;
            return false;
        }
    }
    return true;
}
void error_check(cudaError_t error){
        if(error!=cudaSuccess){
        printf("Error: %s\n",cudaGetErrorString(error));
        exit(-1);
    }
}
unsigned long long int myCPUTimer(unsigned long long int start=0){
    timeval tv;
    gettimeofday(&tv, 0);
    return ((tv.tv_sec*USECPSEC)+tv.tv_usec)-start;
}
int main(int argc, char *argv[]) {


    // TODO: Read input0.raw (matrix A) and input1.raw (matrix B)
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
    std::string result_file = folder + "result.raw";
    std::string reference_file = folder + "output.raw";
    int m, n, p;  // A is m x n, B is n x p, C is m x p



    // TODO Read input0.raw (matrix A)
    ifstream fptr_input0(input0_file);
    fptr_input0>>m>>n;

    float* A = new float[m*n];
    for(int i=0;i<m;i++)
        for(int j=0;j<n;j++)
            fptr_input0>>A[i*n+j];
    // TODO Read input1.raw (matrix B)
    ifstream fptr_input1(input1_file);
    fptr_input1>>n>>p;
    float* B = new float[n*p];
    for(int i=0;i<n;i++)
        for(int j=0;j<p;j++)
            fptr_input1>>B[i*p+j];
    // Allocate memory for result matrices
    float *C_naive = new float[m * p];
    float *C_tiled = new float[m * p];

    // TODO: Use cudaMalloc and cudaMemcpy for GPU memory
    float *A_d,*B_d,*C_d;
    cudaMalloc((void**)&A_d,m*n*sizeof(float));
    cudaMalloc((void**)&B_d,n*p*sizeof(float));
    cudaMalloc((void**)&C_d,m*p*sizeof(float));
    //
    int host_to_device_transfer_time=myCPUTimer();
    cudaMemcpy(A_d,A,m*n*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(B_d,B,n*p*sizeof(float),cudaMemcpyHostToDevice);
    host_to_device_transfer_time=myCPUTimer()-host_to_device_transfer_time;
    // Measure naive CUDA performance
    // TODO: Launch naive_cuda_matmul kernel
    dim3 numThreadsPerBlock(16, 16);
    dim3 numBlocks((p + numThreadsPerBlock.x - 1)/numThreadsPerBlock.x,
    (m + numThreadsPerBlock.y - 1)/numThreadsPerBlock.y+1000);
    int start_time = myCPUTimer();
    naive_cuda_matmul <<< numBlocks, numThreadsPerBlock >>>
    (C_d, A_d, B_d, m, n, p);
    cudaError_t err=cudaDeviceSynchronize();
    int naive_cuda_time = myCPUTimer()-start_time;
    error_check(err);
    // TODO: Write naive CUDA result to file and validate
    // Measure tiled CUDA performance
    
    // Write multiplication result to the file
    int device_to_host_transfer_time=myCPUTimer();
    cudaMemcpy(C_naive,C_d,m*p*sizeof(float),cudaMemcpyDeviceToHost);
    device_to_host_transfer_time=myCPUTimer()-device_to_host_transfer_time;
    ofstream fptr_result(result_file);
    fptr_result.close();
    fptr_result.clear();
    fptr_result.open(result_file);
    fptr_result << m<<" "<<p << endl;
    for(int i=0;i<m;fptr_result<<endl,i++)
        for(int j=0;j<p;j++)
            fptr_result<<C_naive[i*p+j]<<" ";
    // Validate naive result
    bool naive_correct = validate_result(result_file, reference_file);
    if (!naive_correct) {
        std::cerr << "Naive result validation failed for case " << case_number << " in Naive multiplication"<<std::endl;
    }
    // TODO: Launch tiled_cuda_matmul kernel
    start_time = myCPUTimer();
    tiled_cuda_matmul <<< numBlocks, numThreadsPerBlock >>>
    (C_d, A_d, B_d, m, n, p,TILE_WIDTH);
    err=cudaDeviceSynchronize();
    int tiled_cuda_time=myCPUTimer()-start_time;
    error_check(err);
    // TODO: Write tiled CUDA result to file and validate
    cudaMemcpy(C_tiled,C_d,m*p*sizeof(float),cudaMemcpyDeviceToHost);
    fptr_result.close();
    fptr_result.clear();
    fptr_result.open(result_file);
    fptr_result << m<<" "<<p << endl;
    for(int i=0;i<m;fptr_result<<endl,i++)
        for(int j=0;j<p;j++)
            fptr_result<< C_tiled[i*p+j]<<" ";
    // Validate naive result
    bool tiled_correct = validate_result(result_file, reference_file);
    if (!tiled_correct) {
        std::cerr << "Tiled result validation failed for case " << case_number << " in Tiled multiplication"<<std::endl;
    }

    int transfer_time=host_to_device_transfer_time+device_to_host_transfer_time;

    // Print performance results
    std::cout<<"------------------------------\n";
    std::cout << "Case " << case_number << " (" << m << "x" << n << "x" << p << "):\n";
    std::cout<< "Data transfer (from host to device and device to host): "<<(float)transfer_time/1000<< " seconds\n";
    std::cout << "Naive CUDA time: " << (float)naive_cuda_time/1000 << " seconds\n";
    std::cout << "Tiled CUDA time: " << (float)tiled_cuda_time/1000 << " seconds\n";
    std::cout<<"------------------------------\n";
    // Clean up
    delete[] A;
    delete[] B;
    delete[] C_naive;
    delete[] C_tiled;
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    return 0;
}