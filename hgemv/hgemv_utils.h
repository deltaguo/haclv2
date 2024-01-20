#ifndef HGEMV_UTILS_H
#define HGEMV_UTILS_H

#ifndef CAMODEL
#include "kernel_operator.h"
#include "stdio.h"
#endif

using T_INPUT = __fp16;
using T_OUTPUT = float;

constexpr int32_t L0AB_PINGPONG_BUFFER_LEN = 32 * 1024; // 32KB
constexpr int32_t L0C_PINGPONG_BUFFER_LEN = 64 * 1024;  // 64KB
constexpr int32_t L1_PINGPONG_BUFFER_LEN = 128 * 1024;  // 256KB

constexpr int64_t UINT16_STRIDE_LIMIT = 65536;

/**
 * @brief AIC函数：将列优先的nD矩阵转换为nZ格式
 * @param [in] __cbuf__ float *dst：L1 目的地址
 * @param [in] __gm__ float *src: GM 源地址
 * @param [in] int64_t CBUF_M0：在L1中矩阵的行数（至少要32B对齐）
 * @param [in] int64_t CBUF_N0：在L1中矩阵的列数（至少要32B对齐）
 * @param [in] int64_t m_actual：在GM中矩阵的行数（不需要对齐）
 * @param [in] int64_t n_actual：在GM中矩阵的列数（不需要对齐）
 * @param [in] int64_t stride：在GM中矩阵中两列之间的距离（不需要对齐）
 */
__aicore__ __inline__ void ascblas_matrix_gm2cbuf_ND2nZ(
    __cbuf__ __fp16 *dst,
    __gm__ __fp16 *src,
    int64_t CBUF_M0,
    int64_t CBUF_N0,
    int64_t m_actual,
    int64_t n_actual,
    size_t stride)
{
    constexpr int64_t CUBE_K0 = 16;
    if (stride < UINT16_STRIDE_LIMIT)
    {
        copy_gm_to_cbuf_multi_nd2nz_b16(
            dst,
            src,
            static_cast<uint8_t>(0),
            static_cast<uint16_t>(1),
            static_cast<uint16_t>(n_actual),
            static_cast<uint16_t>(m_actual),
            static_cast<uint16_t>(0),
            static_cast<uint16_t>(stride),
            static_cast<uint16_t>(CBUF_N0),
            static_cast<uint16_t>(1),
            static_cast<uint16_t>(0));
    }
    else
    {
        for (int i = 0; i < n_actual; i++)
        {
            copy_gm_to_cbuf_multi_nd2nz_b16(
                dst + i * CUBE_K0,
                src + i * stride,
                static_cast<uint8_t>(0),
                static_cast<uint16_t>(1),
                static_cast<uint16_t>(1),
                static_cast<uint16_t>(m_actual),
                static_cast<uint16_t>(0),
                static_cast<uint16_t>(0),
                static_cast<uint16_t>(CBUF_N0),
                static_cast<uint16_t>(0),
                static_cast<uint16_t>(0));
        }
    }
}

__aicore__ __inline__ void ascblas_gm2l1(__cbuf__ __fp16 *dst,
                                         __gm__ __fp16 *src,
                                         uint16_t n_burst,
                                         uint16_t len_burst,
                                         uint16_t src_stride,
                                         uint16_t dst_stride

)
{
    copy_gm_to_cbuf(dst,
                    src,
                    0,
                    n_burst,
                    len_burst,
                    src_stride - 1,
                    dst_stride - 1,
                    /*pad_t*/ 0);
}

__aicore__ __inline__ void ascblas_l12l0a(__ca__ __fp16 *dst,
                                          __cbuf__ __fp16 *src,
                                          uint8_t repeat,
                                          uint16_t src_stride,
                                          uint16_t dst_stride)
{
    load_cbuf_to_ca(
        dst,
        src,
        0,
        repeat,
        src_stride,
        dst_stride - 1,
        0,
        false,
        inc);
}

__aicore__ __inline__ void ascblas_l12l0a_transpose(__ca__ __fp16 *dst,
                                                    __cbuf__ __fp16 *src,
                                                    uint8_t repeat,
                                                    uint16_t src_stride,
                                                    uint16_t dst_stride)
{
    load_cbuf_to_ca(
        dst,
        src,
        0,
        repeat,
        src_stride,
        dst_stride - 1,
        0,
        true,
        inc);
}

__aicore__ __inline__ void ascblas_l12l0b(__cb__ __fp16 *dst,
                                          __cbuf__ __fp16 *src,
                                          uint8_t repeat,
                                          uint16_t src_stride,
                                          uint16_t dst_stride)
{
    load_cbuf_to_cb(
        dst,
        src,
        0,
        repeat,
        src_stride,
        dst_stride - 1,
        0,
        false,
        inc);
}

__aicore__ __inline__ void ascblas_l12l0b_transpose(__cb__ __fp16 *dst,
                                                    __cbuf__ __fp16 *src,
                                                    uint8_t repeat,
                                                    uint16_t src_stride,
                                                    uint16_t dst_stride)
{
    load_cbuf_to_cb(
        dst,
        src,
        0,
        repeat,
        src_stride,
        dst_stride - 1,
        0,
        true,
        inc);
}

__aicore__ __inline__ void ascblas_l0c2gm(__gm__ float *dst,
                                          __cc__ float *src,
                                          uint16_t vec_dim,
                                          uint16_t vec_num)
{
    copy_matrix_cc_to_gm(
        (__gm__ float*)dst,
        src,
        0,       // sid
        vec_dim, // NSize
        vec_num, // MSize
        16,      // dstStride_dst_D
        16,      // srcStride
        0,       // UnitFlagMode
        F322F16, // QuantPRE
        //NoQuant,
        0,       // ReLUPRE
        false,       // channelSplit
        true     // NZ2ND_EN
    );
}

#endif