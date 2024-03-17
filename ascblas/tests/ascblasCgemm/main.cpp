#include <acl/acl.h>
#include "ascblasCgemm.h"
#include "data_utils.h"

int main(int argc, char** argv)
{
    if (argc < 9 || argc > 11) {
        std::cerr << "Usage: <exe> transA transB M N K lda ldb ldc [verifyLevel] [device_id]" << std::endl;
    }

    ascblasOperation_t transA = num2Op(std::stoi(argv[1]));
    ascblasOperation_t transB = num2Op(std::stoi(argv[2]));
    int64_t M = std::stoi(argv[3]);
    int64_t N = std::stoi(argv[4]);
    int64_t K = std::stoi(argv[5]);
    int64_t lda = std::stoi(argv[6]);
    int64_t ldb = std::stoi(argv[7]);
    int64_t ldc = std::stoi(argv[8]);
    /*
    verifyLevel 表示测试的等级：
    0 表示测试算子性能
    1 表示测试单一输入的正确性
    2 表示测试输入的正确性，并输出误差供csv收集
    */
    int verifyLevel = 0;
    if (argc > 9) verifyLevel = std::stoi(argv[9]);
    int deviceId = 0;
    if (argc > 10) deviceId = std::stoi(argv[10]);
    

    ascComplex alpha {1, 0};
    ascComplex beta {0, 0};

    int64_t A_size = lda * K * sizeof(ascComplex);
    if (transA != ASCBLAS_OP_N) {
        A_size = M * lda * sizeof(ascComplex);
    }
    int64_t B_size = ldb * N * sizeof(ascComplex);
    if (transB != ASCBLAS_OP_N) {
        B_size = K * ldb * sizeof(ascComplex);
    }
    int64_t C_size = ldc * N * sizeof(ascComplex);

    CALL_RT(aclInit(nullptr));

    CALL_RT(aclrtSetDevice(deviceId));


    ascComplex* h_A = nullptr;
    ascComplex* d_A = nullptr;
    if (verifyLevel) CALL_RT(aclrtMallocHost((void**)(&h_A), A_size));
    CALL_RT(aclrtMalloc((void**)(&d_A), A_size, ACL_MEM_MALLOC_HUGE_FIRST));
    if (verifyLevel) ReadFile("../data/A.bin", h_A, A_size);
    if (verifyLevel) CALL_RT(aclrtMemcpy(d_A, A_size, h_A, A_size, ACL_MEMCPY_HOST_TO_DEVICE));

    ascComplex* h_B = nullptr;
    ascComplex* d_B = nullptr;
    if (verifyLevel) CALL_RT(aclrtMallocHost((void**)(&h_B), B_size));
    CALL_RT(aclrtMalloc((void**)(&d_B), B_size, ACL_MEM_MALLOC_HUGE_FIRST));
    if (verifyLevel) ReadFile("../data/B.bin", h_B, B_size);
    if (verifyLevel) CALL_RT(aclrtMemcpy(d_B, B_size, h_B, B_size, ACL_MEMCPY_HOST_TO_DEVICE));

    ascComplex* h_C = nullptr;
    ascComplex* d_C = nullptr;
    if (verifyLevel) CALL_RT(aclrtMallocHost((void**)(&h_C), C_size));
    CALL_RT(aclrtMalloc((void**)(&d_C), C_size, ACL_MEM_MALLOC_HUGE_FIRST));
    if (verifyLevel) ReadFile("../data/C.bin", h_C, C_size);
    if (verifyLevel) CALL_RT(aclrtMemcpy(d_C, C_size, h_C, C_size, ACL_MEMCPY_HOST_TO_DEVICE));

    if (verifyLevel) ReadFile("../data/alpha.bin", &alpha, sizeof(ascComplex));
    if (verifyLevel) ReadFile("../data/beta.bin", &beta, sizeof(ascComplex));

    aclrtStream stream;
    ascblasHandle_t handle;
    ascblasCreate(&handle);
    ascblasGetStream(handle, &stream);

    CALL_RT(ascblasCgemm(
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
    CALL_RT(aclrtSynchronizeStream(stream));
    if (verifyLevel) {
        CALL_RT(aclrtMemcpy(h_C, C_size, d_C, C_size, ACL_MEMCPY_DEVICE_TO_HOST));
        float* h_expect;
        CALL_RT(aclrtMallocHost((void**)(&h_expect), C_size));
        ReadFile("../data/C_expect.bin", h_expect, C_size);
        if (verifyLevel == 1) {
            compareOutputData(reinterpret_cast<float*>(h_C), h_expect, ldc*N*2);
        } else {
            outputError(reinterpret_cast<float*>(h_C), h_expect, ldc*N*2);
        }
        CALL_RT(aclrtFreeHost(h_A));
        CALL_RT(aclrtFreeHost(h_B));
        CALL_RT(aclrtFreeHost(h_C));
    }


    CALL_RT(aclrtFree(d_A));
    CALL_RT(aclrtFree(d_B));
    CALL_RT(aclrtFree(d_C));
    CALL_RT(aclrtResetDevice(deviceId));
    CALL_RT(aclFinalize());

    return 0;
}
