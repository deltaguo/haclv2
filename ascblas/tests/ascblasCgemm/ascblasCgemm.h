#include <fstream>
#include "ascblas.h"
#include <iostream>

aclError ascblasCgemm(
    ascblasHandle_t handle,
    ascblasOperation_t transA, ascblasOperation_t transB,
    int64_t M, int64_t N, int64_t K,
    ascComplex *alpha,
    ascComplex *d_A, int64_t lda,
    ascComplex *d_B, int64_t ldb,
    ascComplex *beta,
    ascComplex *d_C, int64_t ldc
) {
    aclError error;
    aclrtStream stream;
    ascblasGetStream(handle, &stream);
    std::string kernel_name = "ascblasCgemm_kernel";
    std::string bin_name = "ascblasCgemm.o";
    RegisterBinaryKernel(kernel_name.c_str(), bin_name.c_str());

    void* ffts_addr = nullptr;
    uint32_t ffts_len = 0;
    error = rtGetC2cCtrlAddr((uint64_t*)&ffts_addr, &ffts_len);

    int64_t blockDim = 20;
    int64_t pad_num = 128;
    int64_t lda_pad = (transA == ASCBLAS_OP_N)?((M+pad_num-1)/pad_num*pad_num):((K+pad_num-1)/pad_num*pad_num);
    int64_t ldb_pad = (transB == ASCBLAS_OP_N)?((K+pad_num-1)/pad_num*pad_num):((N+pad_num-1)/pad_num*pad_num);
    float* d_A_r = nullptr;
    error = aclrtMalloc((void**)(&d_A_r), (transA == ASCBLAS_OP_N?(lda_pad*K):(lda_pad*M))*sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    float* d_A_i = nullptr;
    error = aclrtMalloc((void**)(&d_A_i), (transA == ASCBLAS_OP_N?(lda_pad*K):(lda_pad*M))*sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    float* d_B_r = nullptr;
    error = aclrtMalloc((void**)(&d_B_r), (transB == ASCBLAS_OP_N?(ldb_pad*N):(ldb_pad*K))*sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    float* d_B_i = nullptr;
    error = aclrtMalloc((void**)(&d_B_i), (transB == ASCBLAS_OP_N?(ldb_pad*N):(ldb_pad*K))*sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    float* d_C_rr = nullptr;
    error = aclrtMalloc((void**)(&d_C_rr), ldc*N*sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    float* d_C_ri = nullptr;
    error = aclrtMalloc((void**)(&d_C_ri), ldc*N*sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    float* d_C_ir = nullptr;
    error = aclrtMalloc((void**)(&d_C_ir), ldc*N*sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    float* d_C_ii = nullptr;
    error = aclrtMalloc((void**)(&d_C_ii), ldc*N*sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    

    typedef struct {
        ascblasOperation_t transA; ascblasOperation_t transB;
        int64_t M; int64_t N; int64_t K;
        ascComplex alpha;
        float *d_A; int64_t lda;
        float *d_B; int64_t ldb;
        ascComplex beta;
        float *d_C; int64_t ldc;
        int64_t lda_pad;
        int64_t ldb_pad;
        float *d_A_r;
        float *d_A_i;
        float *d_B_r;
        float *d_B_i;
        float *d_C_rr;
        float *d_C_ri;
        float *d_C_ir;
        float *d_C_ii;
        void *ffts_addr;
    } KernelArgs;

    KernelArgs kernel_args;

    kernel_args.transA = transA;
    kernel_args.transB = transB;
    kernel_args.M = M;
    kernel_args.N = N;
    kernel_args.K = K;
    kernel_args.alpha = *alpha;
    kernel_args.d_A = reinterpret_cast<float*>(d_A);
    kernel_args.lda = lda;
    kernel_args.d_B = reinterpret_cast<float*>(d_B);
    kernel_args.ldb = ldb;
    kernel_args.beta = *beta;
    kernel_args.d_C = reinterpret_cast<float*>(d_C);
    kernel_args.ldc = ldc;
    kernel_args.lda_pad = lda_pad;
    kernel_args.ldb_pad = ldb_pad;
    kernel_args.d_A_r = d_A_r;
    kernel_args.d_A_i = d_A_i;
    kernel_args.d_B_r = d_B_r;
    kernel_args.d_B_i = d_B_i;
    kernel_args.d_C_rr = d_C_rr;
    kernel_args.d_C_ri = d_C_ri;
    kernel_args.d_C_ir = d_C_ir;
    kernel_args.d_C_ii = d_C_ii;
    kernel_args.ffts_addr = ffts_addr;
    error = rtKernelLaunch((void *)kernel_name.c_str(), blockDim, &kernel_args, sizeof(kernel_args), NULL, stream);

    aclrtSynchronizeStream(stream);
    aclrtFree(d_A_r);
    aclrtFree(d_A_i);
    aclrtFree(d_B_r);
    aclrtFree(d_B_i);
    aclrtFree(d_C_rr);
    aclrtFree(d_C_ri);
    aclrtFree(d_C_ir);
    aclrtFree(d_C_ii);
    return error;
}
