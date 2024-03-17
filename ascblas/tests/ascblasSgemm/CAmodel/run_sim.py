import sys
import numpy as np
from common.ascend_tbe_op import AscendOpKernel, AscendOpKernelRunner

def generate_test_data(batch_size, M, N, K, trans_a=0, trans_b=0):
    matrix_a_list = []
    matrix_b_list = []
    matrix_c_list = []

    for _ in range(batch_size):
        mA = np.random.uniform(-2.0, 2.0, size=(M,K)).astype(np.float32)
        mB = np.random.uniform(-2.0, 2.0, size=(K,N)).astype(np.float32)
        result = np.matmul(mA, mB)
        result = result.astype(np.float32)

        if trans_a:
            mA = mA.transpose([1,0])
            mA = np.ascontiguousarray(mA)
        if trans_b:
            mB = mA.transpose([1,0])
            mB = np.ascontiguousarray(mB)

        matrix_a_list.append(mA)
        matrix_b_list.append(mB)
        matrix_c_list.append(result)

    return np.stack(matrix_a_list), np.stack(matrix_b_list), np.stack(matrix_c_list)

def main():
    # if len(sys.argv[1:]) != 6:
    #     print("Invalid arguments, usage: python3 run_sim.py <Batch> <M> <N> <K> <tarns_a> <trans_b>", file=sys.stderr)
    #     exit(1)
    
    B = 1 
    M = 256
    N = 256
    K = 256
    trans_a = 0
    trans_b = 0
    lda = M
    ldb = K
    ldc = M
    M0 = 128
    N0 = 128
    K0 = 128
    alpha = 1
    beta = 2
    lda_padding = lda
    ldb_padding = ldb
    is_lda_padding = 0
    is_ldb_padding = 0
    is_dot_alpha_add_beta_C =  0
    
    op = AscendOpKernel('../build/ascblasSgemm_ca.o', './ascblasSgemm.json')
    op.need_do_tiling = True
    op.set_output_info([
        {
            'shape': [B, M, N],
            'dtype': 'float32',
            # 'dtype': 'int32',
            'format': 'ND'
        }
    ])
    op_runner = AscendOpKernelRunner(simulator_mode='ca', device_id=0, soc_version='Ascend910B1',
                                     simulator_lib_path='/usr/local/Ascend/ascend-toolkit/latest/tools/simulator')
    
    matrix_a, matrix_b, expected = generate_test_data(B, M, N, K, trans_a, trans_b)
    
    workspace = np.zeros_like(expected)

    matrix_a_padding = np.zeros_like(matrix_a)
    matrix_b_padding = np.zeros_like(matrix_b)



    output_data = op_runner.run(
        op,
        inputs=[np.ones([1, 1]), matrix_a, matrix_b, workspace,  matrix_a_padding,  matrix_b_padding],
        tiling=(np.array([B, trans_a, trans_b, M, N, K, lda, ldb, ldc, M0, N0, K0, alpha, beta, lda_padding, ldb_padding, is_lda_padding, is_ldb_padding, is_dot_alpha_add_beta_C]).astype(np.int32)).tobytes()
    )
    output_data.sync_from_device()
    matrix_c = output_data.get_data()
    # print(matrix_a)
    # print(matrix_b)
    print(matrix_c)
    # print(expected)
    # print(np.allclose(matrix_c, expected, rtol=1e-4, atol=1e-4))
    print([B,M,K,N,trans_a,trans_b])

if __name__ == '__main__':
    main()


# rtGetC2cCtrlAddr

