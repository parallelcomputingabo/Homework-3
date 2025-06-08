#include <iostream>
#include <fstream>
#include <string>
#include <cuda_runtime.h>
#include <omp.h> //Don't know if it makes any sense to use the same stuff for this benchmark
#include <iomanip>
//Because the share float matrix assignment wants a constant value
#define tile_width 16

// Just to make sure
#include <cstdint> 
#include <cmath>
//...

__global__ void naive_cuda_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p) {
    // -TODO-: Implement naive CUDA matrix multiplication
    int idx2 = blockIdx.y * blockDim.y + threadIdx.y;
    int idx1 = blockIdx.x * blockDim.x + threadIdx.x;



    float sum = 0;
    if (idx2 < m && idx1 < p)
    {
        for (int j = 0; j < n; j++)
        {
            sum += A[ idx2*n + j] * B[ j*p + idx1];
        }
        C[idx2 * p + idx1] = sum;
    }
}



__global__ void tiled_cuda_matmul(float *C, float *A, float *B, uint32_t m, uint32_t n, uint32_t p) {
    // -TODO-: Implement tiled CUDA matrix multiplication
    __shared__ float A_t[tile_width][tile_width];
    __shared__ float B_t[tile_width][tile_width];

    int idx2 = blockIdx.y * tile_width + threadIdx.y; //"row"?
    int idx1 = blockIdx.x * tile_width + threadIdx.x; //"col"?


    int x_offset = threadIdx.x;
    int y_offset = threadIdx.y;

    float sum = 0;

    for (int i = 0; i < (n+tile_width-1)/tile_width; i++)
    {
        if (idx2 < m && i*tile_width+x_offset < n)
        {
            A_t[threadIdx.y][threadIdx.x] = A[idx2 * n + i * tile_width + threadIdx.x];
        }
        else
        {
            A_t[threadIdx.y][threadIdx.x]=0.0f;
        }

        //---

        if (idx1 < p && i*tile_width+y_offset < n)
        {
            B_t[threadIdx.y][threadIdx.x] = B[(i * tile_width + threadIdx.y) * p + idx1];
        }
        else
        {
            B_t[threadIdx.y][threadIdx.x]=0.0f;
        }
        __syncthreads();
        for (int j = 0; j < tile_width; j++)
            {
                sum += A_t[y_offset][j] * B_t[j][x_offset];
            }
    __syncthreads();
    }
    if (idx2 < m && idx1 < p) {
        C[idx2 * p + idx1] = sum;
    }


}


bool validate_result(const std::string &result_file, const std::string &reference_file) {
    //-TODO- : Implement result validation *assign2*


    std::ifstream mW(result_file);
    std::ifstream mT(reference_file);


    float test_v, wri_v;
    float rmT, cmT, rmW, cmW;
    bool suc=true;
    mT >> rmT, mT >>cmT, mW >> rmW, mW >> cmW;

    if (rmT!=rmW || cmT!=cmW)
    {
        std::cerr << "Dimension missmatch: ?" <<"\n result: ("<<cmT<<"//"<<cmW<<") output : ("<<rmT<<"//"<<cmT<<")" << std::endl;
        suc=false;
    }

    while(mT >>test_v && mW >> wri_v)
    {
        if (std::round(test_v) != std::round(wri_v) )
        {
            std::cerr << "Error in test: ?" <<"\n The differing values where C: "<< wri_v << " T:"<< test_v << std::endl;
            suc=false;
        }
    }

    return suc;


}

int main(int argc, char *argv[]) {


    // -TODO-: Read input0.raw (matrix A) and input1.raw (matrix B)
    // I'm lazy and took the start bit from the previous exercise
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
    //   ../
    std::string folder = "data/" + std::to_string(case_number) + "/";
    std::string input0_file = folder + "input0.raw";
    std::string input1_file = folder + "input1.raw";
    std::string result_file = folder + "result.raw";
    std::string reference_file = folder + "output.raw";

    // yes
    std::ifstream reference_result(reference_file);
    int m, n, nt, p;


    std::ifstream input0(input0_file);
    input0 >> m >> n;


    std::ifstream input1(input1_file);
    input1 >> nt >> p;


    if (n != nt) {
        std::cerr << "Dimension missmatch!" << std::endl;
        return 1;
    }



    float* A = new float[ m * n ];
    for (uint32_t j = 0; j < m * n; ++j) {
        input0 >> A[j];
    }
    input0.close();     // --TODO-- Read input0.raw (matrix A)



    float* B = new float[ n * p ];
    for (uint32_t j = 0; j < n * p; ++j) {
        input1 >> B[j];
    }
    input1.close();    // --TODO-- Read input1.raw (matrix B)



    float* C = new float[ m * p ];


    // -TODO-: Use cudaMalloc and cudaMemcpy for GPU memory
    int size_A = m*n * sizeof(float);
    int size_B = n*p * sizeof(float);
    int size_C = m*p * sizeof(float);
    float *A_c, *B_c, *C_c;

    cudaMalloc(&A_c, size_A);
    cudaMalloc(&B_c, size_B);
    cudaMalloc(&C_c, size_C);


    int vx=tile_width;
    int vy=tile_width;
    dim3 naive_Block(vx,vy);

    int blockperX = (p + vx - 1) / vx;
    int blockperY = (m + vy - 1) / vy;

    dim3 naive_Grid(blockperX,blockperY);

    // Measure naive CUDA performance
    // -TODO-: Launch naive_cuda_matmul kernel
    double start_time = omp_get_wtime();
    cudaMemcpy(A_c, A, size_A, cudaMemcpyHostToDevice); //Apparently this is to be kept in the measurement
    cudaMemcpy(B_c, B, size_B, cudaMemcpyHostToDevice); //^^
    naive_cuda_matmul<<<naive_Grid, naive_Block>>>(C_c, A_c, B_c, m, n, p);
    cudaMemcpy(C, C_c, size_C, cudaMemcpyDeviceToHost); //^^???
    double naive_cuda_time = omp_get_wtime() - start_time;

    cudaDeviceSynchronize();



    // -TODO-: Write naive CUDA result to file and validate
    // Measure tiled CUDA performance

    std::ofstream output_n(result_file);
    if (!output_n.is_open())
    {
        std::cerr << "Opening result file failed" << std::endl;
    }

    output_n << m << " " << p << std::endl;

    for (int j = 0; j < m; ++j){
        for (int k = 0; k < p; ++k){
            output_n<< std::fixed << std::setprecision(2) << C[j * p + k];
            if (k != p - 1)
            {
                output_n << " ";
            }
        }
        output_n << std::endl;
    }
    output_n.close(); // The autocorrent/fill in in c-lion is crazy, though tit doesn't seem to keep formatting




    // Validate naive result
    bool naive_correct = validate_result(result_file, reference_file);
    if (!naive_correct) {
        std::cerr << "Naive result validation failed for case " << case_number << std::endl;
    }

    //int tile_width = 32;

    dim3 tiled_Block(tile_width, tile_width);
    dim3 tiled_Grid((p+tile_width-1)/tile_width,(m+tile_width-1)/tile_width);


    // -TODO-: Launch tiled_cuda_matmul kernel
    start_time = omp_get_wtime();
    cudaMemcpy(B_c, B, n*p*sizeof(float), cudaMemcpyHostToDevice); // ^^!
    tiled_cuda_matmul<<<tiled_Grid,tiled_Block>>>(C_c,A_c,B_c,m,n,p);
    cudaMemcpy(C, C_c, size_C, cudaMemcpyDeviceToHost); //^^??????
    double tiled_cuda_time = omp_get_wtime() - start_time;

    cudaDeviceSynchronize();


    // -TODO-: Write tiled CUDA result to file and validate
    std::ofstream output_t(result_file);
    if (!output_t.is_open())
    {
        std::cerr << "Opening result file failed" << std::endl;
    }
    output_t << m << " " << p << std::endl;

    for (int j = 0; j < m; ++j){
        for (int k = 0; k < p; ++k){
            output_t<< std::fixed << std::setprecision(2) << C[j * p + k];
            if (k != p - 1)
            {
                output_t << " ";
            }
        }
        output_t << std::endl;
    }
    output_t.close(); // The autocorrent/fill in in c-lion is crazy, though tit doesn't seem to keep formatting




    // Validate naive result
    bool tiled_correct = validate_result(result_file, reference_file);
    if (!tiled_correct) {
        std::cerr << "Tiled result validation failed for case " << case_number << std::endl;
    }



    // Print performance results
    std::cout << "Case " << case_number << " (" << m << "x" << n << "x" << p << "):\n";
    std::cout << "Naive CUDA time: " << naive_cuda_time * 1000 << " milliseconds\n";
    std::cout << "Tiled CUDA time: " << tiled_cuda_time * 1000 << " milliseconds\n";

    // Clean up

    return 0;
}
