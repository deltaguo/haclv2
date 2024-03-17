import sys
import numpy as np
from common.ascend_tbe_op import AscendOpKernel, AscendOpKernelRunner
def gen_random_complex64_numpy(l, r, shape):
    return np.hstack((np.random.uniform(l, r, size=shape),np.random.uniform(l, r, size=shape)))

def gen_golden_data(transA, transB, M, N, K, lda, ldb, ldc):
    l, r = 0, 1
    shape_a = (lda, M) if transA else (lda, K)
    shape_b = (ldb, K) if transB else (ldb, N)
    shape_c = (ldc, N)

    a = gen_random_complex64_numpy(l, r, shape_a)
    b = gen_random_complex64_numpy(l, r, shape_b)
    c = gen_random_complex64_numpy(l, r, shape_c)
    a_r = np.zeros(shape_a)
    a_i = np.zeros(shape_a)
    b_r = np.zeros(shape_b)
    b_i = np.zeros(shape_b)
    c_rr = np.zeros(shape_c)
    c_ri = np.zeros(shape_c)
    c_ir = np.zeros(shape_c)
    c_ii = np.zeros(shape_c)
    return a, b, c, a_r, a_i, b_r, b_i, c_rr, c_ri, c_ir, c_ii
    # return [a], [b], [c], [a_r], [a_i], [b_r], [b_i], [c_rr], [c_ri], [c_ir], [c_ii]

def main():
    M = 256
    N = 256
    K = 256
    trans_a = 0
    trans_b = 0
    lda = M
    ldb = K
    ldc = M
    alpha_a = 1
    alpha_b = 2
    beta_a = 3
    beta_b = 4
    lda_pad = lda
    ldb_pad = ldb

    op = AscendOpKernel('../build/ascblasCgemm_ca.o', './ascblasCgemm.json')
    op.need_do_tiling = True
    op.set_output_info([
        {
            'shape': [M*N*2],
            'dtype': 'float32',
            'format': 'ND'
        }
    ])
    op_runner = AscendOpKernelRunner(simulator_mode='ca', device_id=0, soc_version='Ascend910B1',
                                     simulator_lib_path='/usr/local/Ascend/ascend-toolkit/latest/tools/simulator')
    
    a, b, c, a_r, a_i, b_r, b_i, c_rr, c_ri, c_ir, c_ii = gen_golden_data(0, 0, M, N, K, lda, ldb, ldc)
    output_data = op_runner.run(
        op,
        inputs=[np.ones([1, 1]),a, b, a_r, a_i, b_r, b_i, c_rr, c_ri, c_ir, c_ii],
        tiling=(np.array([trans_a, trans_b, M, N, K, alpha_a, alpha_b, lda, ldb, beta_a, beta_b, ldc, lda_pad, ldb_pad]).astype(np.int32)).tobytes()
    )
    output_data.sync_from_device()
    matrix_c = output_data.get_data()
    # print(matrix_a)
    # print(matrix_b)
    print(matrix_c)
    # print(expected)
    # print(np.allclose(matrix_c, expected, rtol=1e-4, atol=1e-4))
    print([M,K,N,trans_a,trans_b])

if __name__ == '__main__':
    main()


# rtGetC2cCtrlAddr

