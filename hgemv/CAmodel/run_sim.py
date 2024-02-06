import sys
import numpy as np
from common.ascend_tbe_op import AscendOpKernel, AscendOpKernelRunner

def generate_test_data(M, N):
    matrix_a = np.random.uniform(-2.0, 2.0, size=(M,N)).astype(np.float16)
    matrix_b = np.random.uniform(-2.0, 2.0, size=(M,N)).astype(np.float16)
    matrix_c = np.random.uniform(-2.0, 2.0, size=(M,N)).astype(np.float16)

    return matrix_a, matrix_b, matrix_c

def main():
    if len(sys.argv[1:]) != 2:
        print("Invalid arguments, usage: python3 run_sim.py <M> <N> ", file=sys.stderr)
        exit(1)
    
    trans, M, N = [int(arg) for arg in sys.argv[1:]]

    op = AscendOpKernel('../build/hgemv_ca.o', './hgemv.json')
    op.need_do_tiling = True
    op.set_output_info([
        {
            'shape': [M, N],
            'dtype': 'float16',
            'format': 'ND'
        }
    ])
    op_runner = AscendOpKernelRunner(simulator_mode='ca', device_id=0, soc_version='Ascend910B1',
                                     simulator_lib_path='/usr/local/Ascend/ascend-toolkit/latest/tools/simulator')
    
    matrix_a, matrix_b, expected = generate_test_data(M, N)

    output_data = op_runner.run(
        op,
        inputs=[matrix_a, matrix_b],
        tiling=(np.array([trans, M, N, M, 1024, 64, 512, 32, 1]).astype(np.int32)).tobytes()
    )

    output_data.sync_from_device()
    matrix_c = output_data.get_data()
    # print(matrix_a)
    # print(matrix_b)
    # print(matrix_c)
    # print(expected)
    print(np.allclose(matrix_c, expected, rtol=1e-4, atol=1e-4))

if __name__ == '__main__':
    main()


# rtGetC2cCtrlAddr

