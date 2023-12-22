#include "data_utils.h"

#include <acl/acl.h>

extern void matmul_custom_do(uint32_t blockDim, void* l2ctrl, void* stream, uint8_t* a, uint8_t* b, uint8_t* c);

int main()
{
    size_t param1FileSize = 16384 * sizeof(__fp16);
    size_t param2FileSize = 16384 * sizeof(__fp16);
    size_t param3FileSize = 16384 * sizeof(float);
    uint32_t blockDim = 1;

    CALL_RT(aclInit(nullptr));

    int deviceId = 0;
    CALL_RT(aclrtSetDevice(deviceId));

    aclrtStream stream = nullptr;
    CALL_RT(aclrtCreateStream(&stream));

    uint8_t* param1Host;
    uint8_t* param1Device;
    CALL_RT(aclrtMallocHost((void**)(&param1Host), param1FileSize));
    CALL_RT(aclrtMalloc((void**)(&param1Device), param1FileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ReadFile("./input/x1_gm.bin", param1FileSize, param1Host, param1FileSize);
    CALL_RT(aclrtMemcpy(param1Device, param1FileSize, param1Host, param1FileSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* param2Host;
    uint8_t* param2Device;
    CALL_RT(aclrtMallocHost((void**)(&param2Host), param2FileSize));
    CALL_RT(aclrtMalloc((void**)(&param2Device), param2FileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ReadFile("./input/x2_gm.bin", param2FileSize, param2Host, param2FileSize);
    CALL_RT(aclrtMemcpy(param2Device, param2FileSize, param2Host, param2FileSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* param3Host;
    uint8_t* param3Device;
    CALL_RT(aclrtMallocHost((void**)(&param3Host), param3FileSize));
    CALL_RT(aclrtMalloc((void**)(&param3Device), param3FileSize, ACL_MEM_MALLOC_HUGE_FIRST));

    printf("Start to run ...\n");
    matmul_custom_do(blockDim, nullptr, stream, param1Device, param2Device, param3Device);
    CALL_RT(aclrtSynchronizeStream(stream));

    CALL_RT(aclrtMemcpy(param3Host, param3FileSize, param3Device, param3FileSize, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile("./output/output.bin", param3Host, param3FileSize);

    CALL_RT(aclrtFree(param1Device));
    CALL_RT(aclrtFreeHost(param1Host));
    CALL_RT(aclrtFree(param2Device));
    CALL_RT(aclrtFreeHost(param2Host));
    CALL_RT(aclrtFree(param3Device));
    CALL_RT(aclrtFreeHost(param3Host));
    

    CALL_RT(aclrtResetDevice(deviceId));

    CALL_RT(aclFinalize());

    return 0;
}