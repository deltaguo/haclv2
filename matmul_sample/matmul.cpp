#include "kernel_operator.h"

using namespace AscendC;

class KernelMatmul {
public:
    __aicore__ inline KernelMatmul()
    {
        mBlocks = m / 16;
        nBlocks = n / 16;
        kBlocks = k / 16;
    }

    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c)
    {
        aGM.SetGlobalBuffer((__gm__ half*)a);
        bGM.SetGlobalBuffer((__gm__ half*)b);
        cGM.SetGlobalBuffer((__gm__ float*)c);

        pipe.InitBuffer(inQueueA1, 1, L1Size);
        pipe.InitBuffer(inQueueA2, 1, L0ABSize);
        pipe.InitBuffer(inQueueB2, 1, L0ABSize);
        pipe.InitBuffer(outQueueCO1, 1, L0CSize);
    }

    __aicore__ inline void Process()
    {
        LocalTensor<half> L1Local = inQueueA1.AllocTensor<half>();
        LocalTensor<half> aL1Local = L1Local;
        LocalTensor<half> bL1Local = L1Local[128 * 1204];

        LocalTensor<half> aL0Local = inQueueA2.AllocTensor<half>();
        LocalTensor<half> bL0Local = inQueueB2.AllocTensor<half>();
        LocalTensor<float> cL0Local = outQueueCO1.AllocTensor<float>();

        // load matrix A from GM to L1
        Nd2NzParams aDataCopyParams;
        aDataCopyParams.ndNum = 1;
        aDataCopyParams.nValue = m;
        aDataCopyParams.dValue = k;
        aDataCopyParams.srcDValue = k;
        aDataCopyParams.dstNzC0Stride = m;
        aDataCopyParams.dstNzNStride = 1;
        DataCopy(aL1Local, aGM, aDataCopyParams);

        // load matrix B from GM to L1
        Nd2NzParams bDataCopyParams;
        bDataCopyParams.ndNum = 1;
        bDataCopyParams.nValue = k;
        bDataCopyParams.dValue = n;
        bDataCopyParams.srcDValue = n;
        bDataCopyParams.dstNzC0Stride = k;
        bDataCopyParams.dstNzNStride = 1;
        DataCopy(bL1Local, bGM, bDataCopyParams);

        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

        // load matrix A from L1 to L0A
        int srcOffset = 0;
        int dstOffset = 0;
        LoadData2dParams aLoad2dParams;
        aLoad2dParams.repeatTimes = kBlocks;
        aLoad2dParams.srcStride = mBlocks;
        aLoad2dParams.ifTranspose = false;
        for (int i = 0; i < mBlocks; i++) {
            LoadData(aL0Local[dstOffset], aL1Local[srcOffset], aLoad2dParams);
            srcOffset += 16 * 16;
            dstOffset += k * 16;
        }

        // load matrix B from L1 to L0B
        srcOffset = 0;
        dstOffset = 0;
        LoadData2dParams bLoad2dParams;
        bLoad2dParams.repeatTimes = nBlocks;
        bLoad2dParams.srcStride = kBlocks;
        bLoad2dParams.ifTranspose = true;
        for (int i = 0; i < kBlocks; i++) {
            LoadData(bL0Local[dstOffset], bL1Local[srcOffset], bLoad2dParams);
            srcOffset += 16 * 16;
            dstOffset += n * 16;
        }

        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

        MmadParams mmadParams;
        mmadParams.m = m;
        mmadParams.n = n;
        mmadParams.k = k;
        Mmad(cL0Local, aL0Local, bL0Local, mmadParams);

        set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

        FixpipeParams<float> fixpParams;
        fixpParams.burstLen = m * 16 * sizeof(float) / 32;
        fixpParams.dstStride = n;
        fixpParams.srcStride = 0;
        Nz2NdParams nz2ndParams;
        nz2ndParams.nz2ndEn = true;
        nz2ndParams.originalNSize = n;
        fixpParams.nz2ndParams = nz2ndParams;
        fixpParams.quantParams = {QuantMode_t::NoQuant};
        Fixpipe(cGM, cL0Local, fixpParams);
    }

private:
    TPipe pipe;

    TQue<QuePosition::A1, 1> inQueueA1; // L1
    TQue<QuePosition::A2, 1> inQueueA2; // L0A
    TQue<QuePosition::B2, 1> inQueueB2; // L0B
    TQue<QuePosition::CO1, 1> outQueueCO1; // L0C

    GlobalTensor<half> aGM, bGM;
    GlobalTensor<float> cGM;

    uint16_t m = 128;
    uint16_t n = 128;
    uint16_t k = 128;

    uint16_t mBlocks, nBlocks, kBlocks;

    uint32_t L1Size = 512 * 1024;
    uint32_t L0ABSize = 64 * 1024;
    uint32_t L0CSize = 128 * 1024;
};

extern "C" __global__ __aicore__ void matmul_custom(GM_ADDR a, GM_ADDR b, GM_ADDR c)
{
    KernelMatmul op;
    op.Init(a, b, c);
    op.Process();
}

void matmul_custom_do(uint32_t blockDim, void* l2ctrl, void* stream, uint8_t* a, uint8_t* b, uint8_t* c)
{
    matmul_custom<<<blockDim, l2ctrl, stream>>>(a, b, c);
}
