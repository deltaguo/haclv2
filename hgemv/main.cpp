#include "data_utils.h"

#include <acl/acl.h>
#include "rt.h"

int isVerify = false;

/**
 * @brief compare actual data and expected data
 * @param [in] float *actualOutputData: actual data
 * @param [in] float *expectedOutputData: expected data
 * @param [in] uint64_t len: data length
 * @return isCorrect
 */
bool compareFp16OutputData(__fp16 *actualOutputData, __fp16 *expectedOutputData, uint64_t len)
{
    double error = 0;
    int64_t errorCount = 0;
    int64_t lastError = -1;
    int64_t continuous = 0;
    int64_t maxContinous = 0;
    int64_t i = 0;
    float ratios[] = {0.0001, 0.0001};
    for (i = 0; i < len; i++)
    {
        float actualOutputItem = *(actualOutputData + i);
        float expectedOutputItem = *(expectedOutputData + i);
        if (i >= 0 && i < 10)
        {
            printf("our: %f, expected: %f\n", actualOutputItem, expectedOutputItem);
        }
        float tmp = std::abs((std::min(expectedOutputItem, actualOutputItem)) * ratios[1]);
        float limitError = tmp;
        if (std::abs(actualOutputItem - expectedOutputItem) > 0.1 && errorCount < 16)
        {
            std::cout << "index: " << i << " sub super 0.1! actual:" << actualOutputItem << ", expected:" << expectedOutputItem << std::endl;
        }
        if (std::abs(actualOutputItem - expectedOutputItem) > limitError)
        {
            errorCount++;
            if (i == lastError + 1)
            {
                continuous++;
            }
            else
            {
                if (maxContinous < continuous)
                {
                    maxContinous = continuous;
                }
                continuous = 1;
            }
            lastError = i;
        }
        error += std::min((float)1, std::abs((actualOutputItem - expectedOutputItem) / std::min(expectedOutputItem, actualOutputItem)));
    }
    error = error / len;
    std::cout << "error: " << error << std::endl;
    if (i == len - 1)
    {
        if (maxContinous < continuous)
        {
            maxContinous = continuous;
        }
    }

    int count = 0;
    if (errorCount >= len * ratios[0] || maxContinous > 16)
    {
        for (i = 0; i < len; i++)
        {
            float actualOutputItem = *(actualOutputData + i);
            float expectedOutputItem = *(expectedOutputData + i);
            float tmp = std::abs(expectedOutputItem * ratios[1]);
            float limitError = tmp;
            if (std::abs(actualOutputItem - expectedOutputItem) > limitError && errorCount < 16)
            {
                std::cout << "index:" << i << " ,cmprlst:" << std::abs((actualOutputItem - expectedOutputItem) / std::min(expectedOutputItem, actualOutputItem)) << " ,actualDataf:" << actualOutputItem << " ,expectedDataf:" << expectedOutputItem << std::endl;
            }
        }
        printf("cmp len:%lu\n", len);
        std::cout << "errorCount:" << errorCount << std::endl;
        printf("ratio:%.10lf\n", (double)errorCount / len);
        return false;
    }
    else
    {
        printf("cmp len:%lu\n", len);
        std::cout << "errorCount:" << errorCount << std::endl;
        printf("ratio:%.10lf\n", (double)errorCount / len);
        return true;
    }
}

int hgemv(
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
    float *workspace,
    __fp16 *y,
    int incy,
    void* ffts_addr);

int main(int argc, char **argv)
{
    if (argc != 9 && argc != 10)
    {
        std::cerr << "Usage: <exe> trans M N lda incx incy [isVerify]" << std::endl;
        return 0;
    }
    int32_t trans = std::stoi(argv[1]);
    int32_t M = std::stoi(argv[2]);
    int32_t N = std::stoi(argv[3]);
    int32_t lda = std::stoi(argv[4]);
    __fp16 alpha = (__fp16)std::atof(argv[5]);
    __fp16 beta = (__fp16)std::atof(argv[6]);
    int32_t incx = std::stoi(argv[7]);
    int32_t incy = std::stoi(argv[8]);

    if (argc > 9)
        isVerify = std::stoi(argv[9]);

    size_t result_len = trans ? N : M; 

    size_t matrixA_FileSize = lda * N * sizeof(__fp16);
    size_t vectorX_FileSize = N * incx * sizeof(__fp16);
    size_t vectorY_FileSize = M * incy * sizeof(float);
    if (trans)
    {
        matrixA_FileSize = lda * N * sizeof(__fp16);
        vectorX_FileSize = M * incx * sizeof(__fp16);
        vectorY_FileSize = N * incy * sizeof(float);
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
    float *workspace = nullptr;
    if (isVerify)
        CALL_RT(aclrtMallocHost((void **)(&vectorY_Host), vectorY_FileSize));
    CALL_RT(aclrtMalloc((void **)(&vectorY_Device), vectorY_FileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CALL_RT(aclrtMalloc((void **)(&workspace), result_len * incy * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    if (isVerify)
        ReadFile("./data/vectorY.bin", vectorY_FileSize, vectorY_Host, vectorY_FileSize);
    if (isVerify){
        CALL_RT(aclrtMemcpy(vectorY_Device, vectorY_FileSize, vectorY_Host, vectorY_FileSize, ACL_MEMCPY_HOST_TO_DEVICE));
    }

    // vector R
    __fp16 *vectorR = nullptr;
    if (isVerify)
    {
        CALL_RT(aclrtMallocHost((void **)(&vectorR), vectorY_FileSize));
        ReadFile("./data/vectorR.bin", vectorY_FileSize, vectorR, vectorY_FileSize);
    }

    void* ffts_addr;
    uint32_t ffts_len;

    CALL_RT(rtGetC2cCtrlAddr((uint64_t*)&ffts_addr, &ffts_len));
    // for (int i = 0; i < lda; ++i)
    // {
    //     for (int j = 0; j < N; ++j)
    //     {
    //         printf("%.2f ", matrixA_Host[j * lda + i]);
    //     }
    //     printf("\n");
    // }
    printf("Start to run ...\n");
    int ret = hgemv(stream,
                    trans,
                    M,
                    N,
                    &alpha,
                    matrixA_Device,
                    lda,
                    vectorX_Device,
                    incx,
                    &beta,
                    workspace,
                    vectorY_Device,
                    incy,
                    ffts_addr);
    CALL_RT(aclrtSynchronizeStream(stream));
    if(ret){
        exit(1);
    }
    if (isVerify)
    {
        CALL_RT(aclrtMemcpy(vectorY_Host, result_len * incy * sizeof(__fp16), vectorY_Device, result_len * incy * sizeof(__fp16), ACL_MEMCPY_DEVICE_TO_HOST));
        // CALL_RT(aclrtMemcpy(matrixA_Host, result_len * incy * sizeof(float), workspace, result_len * incy * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));
        // printf("A: \n");
        // for(int i = 0;i<512;++i){
        //     printf("%f ", ((float*)matrixA_Host)[i]);
        // }
        // printf("\n");
        // for (int i = 0; i < lda; ++i)
        // {
        //     for (int j = 0; j < N; ++j)
        //     {
        //         printf("%.2f ", matrixA_Host[j * lda + i]);
        //     }
        //     printf("\n");
        //     printf("\n");
        // }
        // for (int i = 0; i < 16; ++i)
        // {
        //     for (int j = 0; j < 16; ++j)
        //     {
        //         printf("%.2f ", matrixA_Host[j * 16 + i]);
        //     }
        //     printf("\n");
        //     printf("\n");
        // }

        // printf("\nX: \n");
        // for (int j = 0; j < M * incx; ++j)
        // {
        //     printf("%.2f ", vectorX_Host[j]);
        // }
        // std::cout << std::endl;
        // printf("Y: \n");
        // for (int j = 0; j < result_len * incy; ++j)
        // {
        //     printf("%.2f ", ((float*)vectorY_Host)[j]);
        // }
        // std::cout << std::endl;
        // printf("R: \n");
        // for (int j = 0; j < result_len * incy; ++j)
        // {
        //     printf("%.2f ", vectorR[j]);
        // }
        // std::cout << std::endl;
        // __fp16* vectorY_fp16 =  new __fp16[result_len*incy];
        // for(int i = 0;i<result_len*incy;++i){
        //     vectorY_fp16[i] = ((float*)vectorY_Host)[i];
        // }
        if (compareFp16OutputData(vectorY_Host, vectorR, result_len * incy))
        {
            std::cout << "correct!" << std::endl;
        }
    }

    CALL_RT(aclrtFree(matrixA_Device));
    CALL_RT(aclrtFree(vectorX_Device));
    CALL_RT(aclrtFree(vectorY_Device));
    CALL_RT(aclrtResetDevice(deviceId));
    CALL_RT(aclFinalize());

    return 0;
}