#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <cstdlib>
#include <sstream>

#define TILE_WIDTH 16


__global__ void naive_cuda_matmul(float* C,const float* A,const float* B,
                                  uint32_t m,uint32_t n,uint32_t p)
{
    uint32_t row = blockIdx.y*blockDim.y + threadIdx.y;
    uint32_t col = blockIdx.x*blockDim.x + threadIdx.x;
    if(row>=m || col>=p) return;

    float acc = 0.f;
    for(uint32_t k=0;k<n;++k)
        acc += A[row*n+k] * B[k*p+col];
    C[row*p+col] = acc;
}

template<int TILE>
__global__ void tiled_cuda_matmul(float* C,const float* A,const float* B,
                                  uint32_t m,uint32_t n,uint32_t p)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    uint32_t row = blockIdx.y*TILE + threadIdx.y;
    uint32_t col = blockIdx.x*TILE + threadIdx.x;
    float acc = 0.f;

    for(uint32_t ph=0; ph<(n+TILE-1)/TILE; ++ph){
        uint32_t aCol = ph*TILE + threadIdx.x;
        uint32_t bRow = ph*TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] =
            (row<m && aCol<n) ? A[row*n + aCol] : 0.f;
        Bs[threadIdx.y][threadIdx.x] =
            (bRow<n && col<p) ? B[bRow*p + col] : 0.f;

        __syncthreads();
        #pragma unroll
        for(int k=0;k<TILE;++k)
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if(row<m && col<p) C[row*p+col] = acc;
}


static void ck(cudaError_t e,const char* where){
    if(e==cudaSuccess) return;
    std::cerr<<"CUDA error @ "<<where<<": "<<cudaGetErrorString(e)<<"\n";
    std::exit(1);
}

struct Matrix {
    uint32_t rows{}, cols{};
    std::vector<float> data;
};

Matrix load_matrix_txt(const std::string& path){
    std::ifstream in(path);
    if(!in) { std::cerr<<"Cannot open "<<path<<"\n"; std::exit(1); }
    Matrix M;
    in >> M.rows >> M.cols;
    M.data.resize(static_cast<size_t>(M.rows)*M.cols);
    for (float& v : M.data) in >> v;
    return M;
}

void save_matrix_txt(const std::string& path,
                     uint32_t rows,uint32_t cols,
                     const std::vector<float>& buf)
{
    std::ofstream out(path);
    out<<rows<<" "<<cols<<"\n";
    out.setf(std::ios::fixed); out.precision(2);
    for(uint32_t i=0;i<rows;++i){
        for(uint32_t j=0;j<cols;++j)
            out<<buf[i*cols+j]<<(j==cols-1?'\n':' ');
    }
}

bool compare_txt(const std::string& gen,const std::string& ref,float eps=1e-3f){
    std::ifstream g(gen), r(ref);
    if(!g||!r) return false;
    uint32_t gm,gp,rm,rp;
    g>>gm>>gp; r>>rm>>rp;
    if(gm!=rm || gp!=rp) return false;
    float vg,vr;
    for(size_t i=0;i<static_cast<size_t>(gm)*gp;++i){
        g>>vg; r>>vr;
        if(std::fabs(vg-vr) > eps*(std::fabs(vr)+1)) return false;
    }
    return true;
}


int main(int argc,char** argv)
{
    if(argc!=2){ std::cerr<<"Usage: "<<argv[0]<<" <case>\n"; return 1; }
    int case_id = std::stoi(argv[1]);
    std::string base = "data/" + std::to_string(case_id) + "/";

    Matrix A = load_matrix_txt(base+"input0.raw");
    Matrix B = load_matrix_txt(base+"input1.raw");
    if(A.cols!=B.rows){
        std::cerr<<"Dimension mismatch\n"; return 1;
    }
    uint32_t m=A.rows, n=A.cols, p=B.cols;

    std::vector<float> C_naive(m*p), C_tiled(m*p);
    float *dA,*dB,*dC;
    ck(cudaMalloc(&dA,A.data.size()*sizeof(float)),"malloc A");
    ck(cudaMalloc(&dB,B.data.size()*sizeof(float)),"malloc B");
    ck(cudaMalloc(&dC,C_naive.size()*sizeof(float)),"malloc C");

    ck(cudaMemcpy(dA,A.data.data(),A.data.size()*sizeof(float),
                  cudaMemcpyHostToDevice),"copy A");
    ck(cudaMemcpy(dB,B.data.data(),B.data.size()*sizeof(float),
                  cudaMemcpyHostToDevice),"copy B");

    dim3 block(TILE_WIDTH,TILE_WIDTH);
    dim3 grid((p+TILE_WIDTH-1)/TILE_WIDTH,(m+TILE_WIDTH-1)/TILE_WIDTH);

    cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    float t_naive=0,t_tiled=0;

    cudaEventRecord(t0);
    naive_cuda_matmul<<<grid,block>>>(dC,dA,dB,m,n,p);
    ck(cudaGetLastError(),"launch naive");
    ck(cudaDeviceSynchronize(),"sync naive");
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    cudaEventElapsedTime(&t_naive,t0,t1);
    ck(cudaMemcpy(C_naive.data(),dC,C_naive.size()*sizeof(float),
                  cudaMemcpyDeviceToHost),"D2H naive");

    cudaEventRecord(t0);
    tiled_cuda_matmul<TILE_WIDTH><<<grid,block>>>(dC,dA,dB,m,n,p);
    ck(cudaGetLastError(),"launch tiled");
    ck(cudaDeviceSynchronize(),"sync tiled");
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    cudaEventElapsedTime(&t_tiled,t0,t1);
    ck(cudaMemcpy(C_tiled.data(),dC,C_tiled.size()*sizeof(float),
                  cudaMemcpyDeviceToHost),"D2H tiled");

    save_matrix_txt(base+"result_naive.raw",m,p,C_naive);
    save_matrix_txt(base+"result_tiled.raw",m,p,C_tiled);
    bool ok_naive  = compare_txt(base+"result_naive.raw", base+"output.raw");
    bool ok_tiled  = compare_txt(base+"result_tiled.raw", base+"output.raw");

    std::cout<<std::fixed<<std::setprecision(5);
    std::cout<<"Case "<<case_id<<" ("<<m<<"x"<<n<<"x"<<p<<")\n";
    std::cout<<"  Naive  : "<<t_naive/1000.0<<" s  "<<(ok_naive?"OK":"FAIKL")<<"\n";
    std::cout<<"  Tiled  : "<<t_tiled/1000.0<<" s  "<<(ok_tiled?"OK":"FAIL")<<"\n";
    if(ok_tiled) std::cout<<"  Speed-up: "<<t_naive/t_tiled<<"\n";

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
