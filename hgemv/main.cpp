#include "data_utils.h"

#include <acl/acl.h>

int isVerify = false;

void hgemv(
    void *stream,
    int trans,
    int M,
    int N,
    const __fp16 *alpha,
    const __fp16 *A,
    int lda,
    const __fp16 *x,
    int incx,
    const __fp16 *beta,
    __fp16 *y,
    int incy);

int main(int argc, char **argv)
{
    if (argc != 7 && argc != 8)
    {
        std::cerr << "Usage: <exe> trans M N lda incx incy [isVerify]" << std::endl;
        return 0;
    }
    int32_t trans = std::stoi(argv[1]);
    int32_t M = std::stoi(argv[2]);
    int32_t N = std::stoi(argv[3]);
    int32_t lda = std::stoi(argv[4]);
    int32_t incx = std::stoi(argv[5]);
    int32_t incy = std::stoi(argv[6]);

    if (argc > 7)
        isVerify = std::stoi(argv[7]);

    size_t matrixA_FileSize = lda * N * sizeof(__fp16);
    size_t vectorX_FileSize = N * incx * sizeof(__fp16);
    size_t vectorY_FileSize = M * incy * sizeof(__fp16);
    if (trans)
    {
        matrixA_FileSize = lda * M * sizeof(__fp16);
        vectorX_FileSize = M * incx * sizeof(__fp16);
        vectorY_FileSize = N * incy * sizeof(__fp16);
    }
    CALL_RT(aclInit(nullptr));
    int deviceId = 0;
    CALL_RT(aclrtSetDevice(deviceId));

    aclrtStream stream = nullptr;
    CALL_RT(aclrtCreateStream(&stream));
    // matrix A
    __fp16 *matrixA_Host = nullptr;
    __fp16 *matrixA_Device = nullptr;
    if (isVerify)
        CALL_RT(aclrtMallocHost((void **)(&matrixA_Host), matrixA_FileSize));
    CALL_RT(aclrtMalloc((void **)(&matrixA_Device), matrixA_FileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    if (isVerify)
        ReadFile("./data/matrixA.bin", matrixA_FileSize, matrixA_Host, matrixA_FileSize);
    if (isVerify)
        CALL_RT(aclrtMemcpy(matrixA_Device, matrixA_FileSize, matrixA_Host, matrixA_FileSize, ACL_MEMCPY_HOST_TO_DEVICE));
    // vector X
    __fp16 *vectorX_Host = nullptr;
    __fp16 *vectorX_Device = nullptr;
    if (isVerify)
        CALL_RT(aclrtMallocHost((void **)(&vectorX_Host), vectorX_FileSize));
    CALL_RT(aclrtMalloc((void **)(&vectorX_Device), vectorX_FileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    if (isVerify)
        ReadFile("./data/vectorX.bin", vectorX_FileSize, vectorX_Host, vectorX_FileSize);
    if (isVerify)
        CALL_RT(aclrtMemcpy(vectorX_Device, vectorX_FileSize, vectorX_Host, vectorX_FileSize, ACL_MEMCPY_HOST_TO_DEVICE));
    // vector Y
    __fp16 *vectorY_Host = nullptr;
    __fp16 *vectorY_Device = nullptr;
    if (isVerify)
        CALL_RT(aclrtMallocHost((void **)(&vectorY_Host), vectorY_FileSize));
    CALL_RT(aclrtMalloc((void **)(&vectorY_Device), vectorY_FileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    if (isVerify)
        ReadFile("./data/vectorY.bin", vectorY_FileSize, vectorY_Host, vectorY_FileSize);
    if (isVerify)
        CALL_RT(aclrtMemcpy(vectorY_Device, vectorY_FileSize, vectorY_Host, vectorY_FileSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // vector R
    __fp16 *vectorR = nullptr;
    if (isVerify)
    {
        CALL_RT(aclrtMallocHost((void **)(&vectorR), vectorY_FileSize));
        ReadFile("./data/vectorR.bin", vectorY_FileSize, vectorR, vectorY_FileSize);
    }

    __fp16 alpha = 1.0;
    __fp16 beta = 1.0;
    printf("Start to run ...\n");
    hgemv(stream,
          trans,
          M,
          N,
          &alpha,
          matrixA_Device,
          lda,
          vectorX_Device,
          incx,
          &beta,
          vectorY_Device,
          incy);

    printf("A: \n");
    for (int j = 0; j < lda * N; ++j)
    {
        printf("%.2f ", matrixA_Host[j]);
    }
    printf("X: \n");
    for (int j = 0; j < N * incx; ++j)
    {
        printf("%.2f ", vectorX_Host[j]);
    }
    std::cout << std::endl;
    printf("Y: \n");
    for (int j = 0; j < M * incy; ++j)
    {
        printf("%.2f ", vectorY_Host[j]);
    }
    std::cout << std::endl;
    printf("R: \n");
    for (int j = 0; j < M * incy; ++j)
    {
        printf("%.2f ", vectorR[j]);
    }
    std::cout << std::endl;
    CALL_RT(aclrtFree(matrixA_Device));
    CALL_RT(aclrtFree(vectorX_Device));
    CALL_RT(aclrtFree(vectorY_Device));
    CALL_RT(aclrtResetDevice(deviceId));
    CALL_RT(aclFinalize());

    return 0;
}