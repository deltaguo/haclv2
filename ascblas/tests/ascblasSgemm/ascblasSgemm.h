#include <fstream>
#include "ascblas.h"

aclError ascblasSgemm(
    ascblasHandle_t handle,
    ascblasOperation_t transA,
    ascblasOperation_t transB,
    int64_t M,
    int64_t N,
    int64_t K,
    float *alpha,
    float *d_A,
    int64_t lda,
    float *d_B,
    int64_t ldb,
    float *beta,
    float *d_C,
    int64_t ldc
) {
    aclError error;
    aclrtStream stream;
    // 得到stream
    ascblasGetStream(handle, &stream);
    // 注册内核函数
    std::string kernel_name = "ascblasSgemm_kernel";
    std::string bin_name = "ascblasSgemm.o";
    RegisterBinaryKernel(kernel_name.c_str(), bin_name.c_str());

    // 声明AIV，AIC同步的变量
    void* ffts_addr = nullptr;
    uint32_t ffts_len = 0;
    error = rtGetC2cCtrlAddr((uint64_t*)&ffts_addr, &ffts_len);

    // 按照输入数据计算分块大小
    int64_t M_round = (M + 16 - 1) / 16 * 16;
    int64_t N_round = (N + 16 - 1) / 16 * 16;
    int64_t K_round = (K + 16 - 1) / 16 * 16;
    int64_t M0 = 128;
    int64_t N0 = 128;
    int64_t K0 = 128;
    int64_t M0_tile_num_of_M = (M + M0 - 1) / M0;
    int64_t N0_tile_num_of_N = (N + N0 - 1) / N0;
    int64_t groupDim = M0_tile_num_of_M * N0_tile_num_of_N;
    // groupDim 表示启用的物理核数,不要大于CORENUM
    groupDim = groupDim < CORENUM ? groupDim : CORENUM;
    // groupDim = 1;

    // 对于 lda, ldb 不满足512B对齐的情况，需要padding
    int64_t padding_num = 128;
    // 按参与计算的矩阵大小来分配padding后的lda
    int lda_padding;
    if (transA == ASCBLAS_OP_T) {
        lda_padding = (K + padding_num - 1) / padding_num * padding_num;
    } else {
        lda_padding = (M + padding_num - 1) / padding_num * padding_num;
    }
    int ldb_padding;
    if (transB == ASCBLAS_OP_T) {
        ldb_padding = (N + padding_num - 1) / padding_num * padding_num;
    } else {
        ldb_padding = (K + padding_num - 1) / padding_num * padding_num;
    }
    
    // AIV,AIC数据的交换的空间
    float* AIV_AIC_workspace = nullptr;
    error = aclrtMalloc((void**)(&AIV_AIC_workspace), 2 * groupDim * (M0 * N0) * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    
    float* gm_A_padding = nullptr;
    float* gm_B_padding = nullptr;
    // 是否lda, ldb 需要padding
    int64_t is_lda_padding = true;
    int64_t is_ldb_padding = true;


    // 需要padding则分配额外空间
    int64_t A_byte;
    if (transA == ASCBLAS_OP_T) {
        A_byte = (M + M0 - 1) / M0 * M0 * lda_padding * sizeof(float);
    } else {
        A_byte = lda_padding * (K + K0 - 1) / K0 * K0 * sizeof(float);
    }
    error = aclrtMalloc((void**)(&gm_A_padding), A_byte, ACL_MEM_MALLOC_HUGE_FIRST);

    int64_t B_byte;
    if (transB == ASCBLAS_OP_T) {
        B_byte = (K + K0 - 1) / K0 * K0 * ldb_padding * sizeof(float);
    } else {
        B_byte = ldb_padding * (N + N0 - 1) / N0 * N0 * sizeof(float);
    }
    error = aclrtMalloc((void**)(&gm_B_padding), B_byte, ACL_MEM_MALLOC_HUGE_FIRST);


    // 编写内核函数结构体
    typedef struct {
        ascblasOperation_t transA;
        ascblasOperation_t transB;
        int64_t M;
        int64_t N;
        int64_t K;
        float alpha;
        float *gm_A;
        int64_t lda;
        float *gm_B;
        int64_t ldb;
        float beta;
        float *gm_C;
        int64_t ldc;
        int64_t batchSize;
        int64_t M0;
        int64_t N0;
        int64_t K0;
        float *AIV_AIC_workspace;
        void *ffts_addr;
        float *gm_A_padding;
        int64_t lda_padding;
        float *gm_B_padding;
        int64_t ldb_padding;
        int64_t is_lda_padding;
        int64_t is_ldb_padding;
        int64_t is_dot_alpha_add_beta_C;
    } KernelArgs;

    KernelArgs kernel_args;

    kernel_args.transA = transA;
    kernel_args.transB = transB;
    kernel_args.M = M;
    kernel_args.N = N;
    kernel_args.K = K;
    kernel_args.alpha = *alpha;
    kernel_args.gm_A = d_A;
    kernel_args.lda = lda;
    kernel_args.gm_B = d_B;
    kernel_args.ldb = ldb;
    kernel_args.beta = *beta;
    kernel_args.gm_C = d_C;
    kernel_args.ldc = ldc;
    kernel_args.batchSize = 1;
    kernel_args.M0 = M0;
    kernel_args.N0 = N0;
    kernel_args.K0 = K0;
    kernel_args.AIV_AIC_workspace = AIV_AIC_workspace;
    kernel_args.ffts_addr = ffts_addr;
    kernel_args.gm_A_padding = gm_A_padding;
    kernel_args.lda_padding = lda_padding;
    kernel_args.gm_B_padding = gm_B_padding;
    kernel_args.ldb_padding = ldb_padding;
    kernel_args.is_lda_padding = is_lda_padding;
    kernel_args.is_ldb_padding = is_ldb_padding;
    kernel_args.is_dot_alpha_add_beta_C = !(*alpha == 1.0f && *beta == 0.0f);
    
    // 调用核函数
    error = rtKernelLaunch((void *)kernel_name.c_str(), groupDim, &kernel_args, sizeof(kernel_args), NULL, stream);

    // aclrtSynchronizeStream(stream);
    // float* h_A_padding = nullptr;
    // aclrtMallocHost((void**)(&h_A_padding), A_byte);
    // aclrtMemcpy(h_A_padding, A_byte, gm_A_padding, A_byte, ACL_MEMCPY_DEVICE_TO_HOST);

    // printf("hA_padding: \n");
    // for (int i = 0; i < 128; i++) {
    //     for (int j = 0; j < 512; j++) {
    //         printf("%d ", (int)h_A_padding[i + j * 128]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");
    

    // float* h_B_padding = nullptr;
    // aclrtMallocHost((void**)(&h_B_padding), B_byte);
    // aclrtMemcpy(h_B_padding, B_byte, gm_B_padding, B_byte, ACL_MEMCPY_DEVICE_TO_HOST);

    // printf("hB_padding: \n");
    // for (int i = 0; i < ldb; i++) {
    //     for (int j = 0; j < N; j++) {
    //         printf("%d ", (int)h_B_padding[i + j * ldb]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");

    return error;
}
