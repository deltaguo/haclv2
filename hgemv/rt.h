#ifndef __MIX_KERNEL_DEMO_RT_H__
#define __MIX_KERNEL_DEMO_RT_H__

#ifdef __cplusplus
extern "C" {
#endif

typedef void* rtStream_t;

typedef enum tagRtError {
    RT_ERROR_NONE = 0x0, // success
    // ...
} rtError_t;

#define RT_DEV_BINARY_MAGIC_ELF 0x43554245

typedef struct tagRtDevBinary {
    uint32_t magic;
    uint32_t version;
    const void* data;
    uint64_t length;
} rtDevBinary_t;

rtError_t rtDevBinaryRegister(
    const rtDevBinary_t* bin,
    void** handle
);

rtError_t rtKernelLaunch(
    const void* stubFunc,
    uint32_t blockDim,
    void* args,
    uint32_t argsSize,
    void* smDesc,
    rtStream_t stream
);

rtError_t rtFunctionRegister(
    void* binHandle,
    const void* stubFunc,
    const char* stubName,
    const void* devFunc,
    uint32_t funcMode
);

rtError_t rtGetC2cCtrlAddr(uint64_t* addr, uint32_t* len);

#ifdef __cplusplus
}
#endif

#endif // __MIX_KERNEL_DEMO_RT_H__