#include "data_utils.h"

#include <acl/acl.h>

int isVerify = false;

void asc_ppmatmul(
    void* stream,
    float* gm_A,
    float* gm_B,
    float* gm_C,
    int32_t batchSize,
    int trans_a,
    int trans_b,
    int32_t M,
    int32_t N,
    int32_t K,
    int32_t lda,
    int32_t ldb,
    int32_t ldc
);


int main(int argc, char** argv)
{
    if (argc != 10 && argc != 11) {
        std::cerr << "Usage: <exe> batch trans_a trans_b M K N lda ldb ldc [isVerify]" << std::endl;
    }
    int32_t batch = std::stoi(argv[1]);
    int trans_a = std::stoi(argv[2]);
    int trans_b = std::stoi(argv[3]);
    int32_t M = std::stoi(argv[4]);
    int32_t N = std::stoi(argv[5]);
    int32_t K = std::stoi(argv[6]);
    int32_t lda = std::stoi(argv[7]);
    int32_t ldb = std::stoi(argv[8]);
    int32_t ldc = std::stoi(argv[9]);

    if (argc > 10) isVerify = std::stoi(argv[10]);

    size_t matrixA_FileSize = batch * M * lda * sizeof(float);
    if (trans_a) {
        matrixA_FileSize = batch * K * lda * sizeof(float);
    }
    size_t matrixB_FileSize = batch * K * ldb * sizeof(float);
    if (trans_b) {
        matrixB_FileSize = batch * N * ldb * sizeof(float);
    }
    size_t matrixC_FileSize = batch * M * ldc * sizeof(float);

    CALL_RT(aclInit(nullptr));

    int deviceId = 0;
    CALL_RT(aclrtSetDevice(deviceId));

    aclrtStream stream = nullptr;
    CALL_RT(aclrtCreateStream(&stream));

    float* matrixA_Host = nullptr;
    float* matrixA_Device = nullptr;
    if (isVerify) CALL_RT(aclrtMallocHost((void**)(&matrixA_Host), matrixA_FileSize));
    CALL_RT(aclrtMalloc((void**)(&matrixA_Device), matrixA_FileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    if (isVerify) ReadFile("./data/matrixA.bin", matrixA_FileSize, matrixA_Host, matrixA_FileSize);
    if (isVerify) CALL_RT(aclrtMemcpy(matrixA_Device, matrixA_FileSize, matrixA_Host, matrixA_FileSize, ACL_MEMCPY_HOST_TO_DEVICE));

    float* matrixB_Host = nullptr;
    float* matrixB_Device = nullptr;
    if (isVerify) CALL_RT(aclrtMallocHost((void**)(&matrixB_Host), matrixB_FileSize));
    CALL_RT(aclrtMalloc((void**)(&matrixB_Device), matrixB_FileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    if (isVerify) ReadFile("./data/matrixB.bin", matrixB_FileSize, matrixB_Host, matrixB_FileSize);
    if (isVerify) CALL_RT(aclrtMemcpy(matrixB_Device, matrixB_FileSize, matrixB_Host, matrixB_FileSize, ACL_MEMCPY_HOST_TO_DEVICE));

    float* matrixC_Host = nullptr;
    float* matrixC_Device = nullptr;
    if (isVerify) CALL_RT(aclrtMallocHost((void**)(&matrixC_Host), matrixC_FileSize));
    CALL_RT(aclrtMalloc((void**)(&matrixC_Device), matrixC_FileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    if (isVerify) ReadFile("./data/matrixC.bin", matrixC_FileSize, matrixC_Host, matrixC_FileSize);
    if (isVerify) CALL_RT(aclrtMemcpy(matrixC_Device, matrixC_FileSize, matrixC_Host, matrixC_FileSize, ACL_MEMCPY_HOST_TO_DEVICE));

    printf("Start to run ...\n");
    asc_ppmatmul(
        stream,
        matrixA_Device,
        matrixB_Device,
        matrixC_Device,
        batch,
        trans_a,
        trans_b,
        M,
        N,
        K,
        lda,
        ldb,
        ldc
    );
    // matmul_custom_do(blockDim, nullptr, stream, matrixA_Device, matrixB_Device, matrixC_Device);
    CALL_RT(aclrtSynchronizeStream(stream));
    if (isVerify) {
        CALL_RT(aclrtMemcpy(matrixC_Host, matrixC_FileSize, matrixC_Device, matrixC_FileSize, ACL_MEMCPY_DEVICE_TO_HOST));
        float* golden_Host;
        CALL_RT(aclrtMallocHost((void**)(&golden_Host), matrixC_FileSize));
        ReadFile("./data/golden.bin", matrixC_FileSize, golden_Host, matrixC_FileSize);
        auto hc = reinterpret_cast<float *>(matrixC_Host);
        auto hg = reinterpret_cast<float *>(golden_Host);
        compareFp32OutputData(hc, hg, batch * M * ldc);
        // printf("hg: \n");
        // for (int i = 0; i < M; i++) {
        //     for (int j = 0; j < ldc; j++) {
        //         printf("%.1f ", (float)hg[i * ldc + j]);
        //     }
        //     std::cout << "\n";
        // }
        // std::cout << "\n";
        // printf("hc: \n");
        // for (int i = 0; i < M; i++) {
        //     for (int j = 0; j < ldc; j++) {
        //         printf("%.1f ", (float)hc[i * ldc + j]);
        //     }
        //     std::cout << "\n";
        // }
        // std::cout << "\n";
        CALL_RT(aclrtFreeHost(matrixA_Host));
        CALL_RT(aclrtFreeHost(matrixB_Host));
        CALL_RT(aclrtFreeHost(matrixC_Host));
    }


    CALL_RT(aclrtFree(matrixA_Device));
    CALL_RT(aclrtFree(matrixB_Device));
    CALL_RT(aclrtFree(matrixC_Device));
    CALL_RT(aclrtResetDevice(deviceId));
    CALL_RT(aclFinalize());

    return 0;
}
