#include <acl/acl.h>
#include "ascblasHgemm.h"
#include "data_utils.h"

/*
verifyLevel 表示测试的等级：
0 表示测试算子性能
1 表示测试单一输入的正确性
2 表示测试输入的正确性，并输出误差供csv收集
*/
int verifyLevel = 0;

// deviceId 表示程序运行在第几号卡上
int deviceId = 0;


int main(int argc, char** argv)
{
    if (argc != 9 && argc != 10 && argc != 11) {
        std::cerr << "Usage: <exe> transA transB M N K lda ldb ldc [verifyLevel] [deviceId]" << std::endl;
    }

    // 获取参数
    ascblasOperation_t transA = (std::stoi(argv[1]) == 0) ? ASCBLAS_OP_N : ASCBLAS_OP_T;
    ascblasOperation_t transB = (std::stoi(argv[2]) == 0) ? ASCBLAS_OP_N : ASCBLAS_OP_T;
    int32_t M = std::stoi(argv[3]);
    int32_t N = std::stoi(argv[4]);
    int32_t K = std::stoi(argv[5]);
    int32_t lda = std::stoi(argv[6]);
    int32_t ldb = std::stoi(argv[7]);
    int32_t ldc = std::stoi(argv[8]);
    if (argc > 9) verifyLevel = std::stoi(argv[9]);
    if (argc > 10) deviceId = std::stoi(argv[10]);

    // printf("transA: %d, transB: %d, M: %d, N: %d, K: %d, lda: %d, ldb: %d, ldc: %d \n", transA, transB, M, N, K, lda, ldb, ldc);

    __fp16 alpha = 1.0f;
    __fp16 beta = 1.0f;

    size_t A_size;
    if (transA == ASCBLAS_OP_T) {
        A_size = M * lda * sizeof(__fp16);
    } else {
        A_size = lda * K * sizeof(__fp16);
    }
    size_t B_size;
    if (transB == ASCBLAS_OP_T) {
        B_size = K * ldb * sizeof(__fp16);
    } else {
        B_size = ldb * N * sizeof(__fp16);
    }
    size_t C_size = ldc * N * sizeof(__fp16);

    CALL_RT(aclInit(nullptr));

    CALL_RT(aclrtSetDevice(deviceId));

    // 调用ascblas算子
    aclrtStream stream;
    ascblasHandle_t handle;
    ascblasCreate(&handle);
    ascblasGetStream(handle, &stream);

    // 分配host，device端内存，并读取数据
    __fp16* h_A = nullptr;
    __fp16* h_B = nullptr;
    __fp16* h_C = nullptr;
    __fp16* d_A = nullptr;
    __fp16* d_B = nullptr;
    __fp16* d_C = nullptr;

    if (verifyLevel) {
        CALL_RT(aclrtMallocHost((void**)(&h_A), A_size));
        CALL_RT(aclrtMallocHost((void**)(&h_B), B_size));
        CALL_RT(aclrtMallocHost((void**)(&h_C), C_size));
        ReadFile("../data/A.bin", h_A, A_size);
        ReadFile("../data/B.bin", h_B, B_size);
        ReadFile("../data/C.bin", h_C, C_size);
        ReadFile("../data/alpha.bin", &alpha, sizeof(__fp16));
        ReadFile("../data/beta.bin", &beta, sizeof(__fp16));
    }

    for (int i = 0; i < (verifyLevel == 0 ? 21 : 1); i++) {
        CALL_RT(aclrtMalloc((void**)(&d_A), A_size, ACL_MEM_MALLOC_HUGE_FIRST));
        CALL_RT(aclrtMalloc((void**)(&d_B), B_size, ACL_MEM_MALLOC_HUGE_FIRST));
        CALL_RT(aclrtMalloc((void**)(&d_C), C_size, ACL_MEM_MALLOC_HUGE_FIRST));

        if (verifyLevel) {
            CALL_RT(aclrtMemcpy(d_A, A_size, h_A, A_size, ACL_MEMCPY_HOST_TO_DEVICE));
            CALL_RT(aclrtMemcpy(d_B, B_size, h_B, B_size, ACL_MEMCPY_HOST_TO_DEVICE));
            CALL_RT(aclrtMemcpy(d_C, C_size, h_C, C_size, ACL_MEMCPY_HOST_TO_DEVICE));
        }

        CALL_RT(ascblasHgemm(
            handle,
            transA,
            transB,
            M,
            N,
            K,
            &alpha,
            d_A,
            lda,
            d_B,
            ldb,
            &beta,
            d_C,
            ldc
        ));

        // 算子结束后需要同步，才能测数据
        CALL_RT(aclrtSynchronizeStream(stream));

        // 如果需要测试结果正确，则和期望值对比
        // verifyLevel == 2 会输出误差让csv获取
        if (verifyLevel) {
            CALL_RT(aclrtMemcpy(h_C, C_size, d_C, C_size, ACL_MEMCPY_DEVICE_TO_HOST));
            __fp16* h_expect;
            CALL_RT(aclrtMallocHost((void**)(&h_expect), C_size));
            ReadFile("../data/C_expect.bin", h_expect, C_size);
            if (verifyLevel == 1) compareOutputData(h_C, h_expect, ldc * N);
            if (verifyLevel == 2) outputError(h_C, h_expect, ldc * N);
            if (verifyLevel == 1) printf("transA: %d, transB: %d, M: %d, N: %d, K: %d, lda: %d, ldb: %d, ldc: %d, alpha: %f, beta: %f \n", transA, transB, M, N, K, lda, ldb, ldc, alpha, beta);

            CALL_RT(aclrtFreeHost(h_A));
            CALL_RT(aclrtFreeHost(h_B));
            CALL_RT(aclrtFreeHost(h_C));
            CALL_RT(aclrtFreeHost(h_expect));
        }
        
        CALL_RT(aclrtFree(d_A));
        CALL_RT(aclrtFree(d_B));
        CALL_RT(aclrtFree(d_C));
    }

    CALL_RT(aclrtResetDevice(deviceId));
    CALL_RT(aclFinalize());

    return 0;
}
