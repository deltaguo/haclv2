import numpy as np
import argparse

def generate_test_data(trans_a, trans_b, M, N, K, lda, ldb, ldc, batch_size=1):
    matrix_a_list = []
    matrix_b_list = []
    matrix_c_list = []
    golden_list = []
    for _ in range(batch_size):
        # mA = np.random.uniform(0, 1, size=(M, K)).astype(np.float32)
        # mB = np.random.uniform(0, 1, size=(K, N)).astype(np.float32)
        # mA = np.random.randint(1, 2, size=(M, K)).astype(np.float32)
        # mB = np.random.randint(1, 2, size=(K, N)).astype(np.float32)
        if (trans_a == 0):
            mA = np.random.randint(-10, 10, size=(M, lda)).astype(np.float32)
        else:
            mA = np.random.randint(-10, 10, size=(K, lda)).astype(np.float32)
        if (trans_b == 0):
            mB = np.random.randint(-10, 10, size=(K, ldb)).astype(np.float32)
        else:
            mB = np.random.randint(-10, 10, size=(N, ldb)).astype(np.float32)
        
        # if (trans_a == 0):
        #     mA = np.random.randint(1, 2, size=(M, lda)).astype(np.float32)
        # else:
        #     mA = np.random.randint(1, 2, size=(K, lda)).astype(np.float32)
        # if (trans_b == 0):
        #     mB = np.random.randint(1, 2, size=(K, ldb)).astype(np.float32)
        # else:
        #     mB = np.random.randint(1, 2, size=(N, ldb)).astype(np.float32)

        mc = np.random.randint(-10, 10, size=(M, ldc)).astype(np.float32)

        matrix_c_list.append(mc)
        # for i in range(M):
        #     for j in range(K):
        #         mA[i][j] = int(j / 8) * 0.1 + int(i / 16)
        # print("mA")
        # print(mA)
        # mB = np.eye(N).astype(np.float32)
        if (trans_a == 0):
            mA_real = np.ascontiguousarray(mA[0:M,0:K])
        else:
            mA_real = mA[0:K,0:M]
            mA_real = mA_real.transpose([1, 0])
            mA_real = np.ascontiguousarray(mA_real)
        if (trans_b == 0):
            mB_real = np.ascontiguousarray(mB[0:K,0:N])
        else:
            mB_real = mB[0:N,0:K]
            mB_real = mB_real.transpose([1, 0])
            mB_real = np.ascontiguousarray(mB_real)
        
        result = np.matmul(mA_real, mB_real)
        result = result.astype(np.float32)

        mc[0:M,0:N] = result

        # if (trans_a):
        #     mA = mA.transpose([1, 0])
        #     mA = np.ascontiguousarray(mA)
        # if (trans_b):
        #     mB = mB.transpose([1, 0])
        #     mB = np.ascontiguousarray(mB)
        
        matrix_a_list.append(mA)
        matrix_b_list.append(mB)
        golden_list.append(mc)
    np.stack(matrix_a_list).tofile(f'data/matrixA.bin')
    np.stack(matrix_b_list).tofile(f'data/matrixB.bin')
    np.stack(matrix_c_list).tofile(f'data/matrixC.bin')
    np.stack(golden_list).tofile(f'data/golden.bin')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', action='store', type=int, default=1)
    parser.add_argument('--trans_a', action='store', type=int, default=0)
    parser.add_argument('--trans_b', action='store', type=int, default=0)
    parser.add_argument('M', action='store', type=int)
    parser.add_argument('N', action='store', type=int)
    parser.add_argument('K', action='store', type=int)
    parser.add_argument('lda', action='store', type=int)
    parser.add_argument('ldb', action='store', type=int)
    parser.add_argument('ldc', action='store', type=int)

    args = parser.parse_args()

    generate_test_data(args.trans_a, args.trans_b, args.M, args.N, args.K, args.lda, args.ldb, args.ldc, args.batch)