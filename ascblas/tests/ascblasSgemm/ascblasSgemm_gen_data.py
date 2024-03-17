import numpy as np
import argparse

def calc_expect_func(transA, transB, M, N, K, lda, ldb, ldc, src_type='float32'):
    shape_a = (lda, K)
    shape_b = (ldb, N)
    shape_c = (ldc, N)

    if transA > 0:
        shape_a = (lda, M)

    if transB > 0:
        shape_b = (ldb, K)
    
    # a = np.random.uniform(-1, 1, size=shape_a[0] * shape_a[1]).astype(np.float32)
    # b = np.random.uniform(-1, 1, size=shape_b[0] * shape_b[1]).astype(np.float32)
    # c = np.random.uniform(-1, 1, size=shape_c[0] * shape_c[1]).astype(np.float32)
    a = np.random.randint(-10, 10, size=shape_a[0] * shape_a[1]).astype(np.float32)
    b = np.random.randint(-10, 10, size=shape_b[0] * shape_b[1]).astype(np.float32)
    c = np.random.randint(-10, 10, size=shape_c[0] * shape_c[1]).astype(np.float32)
    # a = np.ones(shape_a[0] * shape_a[1], dtype=float).astype(np.float32)
    # b = np.ones(shape_b[0] * shape_b[1], dtype=float).astype(np.float32)
    # c = np.ones(shape_c[0] * shape_c[1], dtype=float).astype(np.float32)
    # for i in range(shape_a[0] * shape_a[1]):
    #     a[i] = i // 128
    # for i in range(shape_b[0] * shape_b[1]):
    #     b[i] = i // 128 + 1
    # for i in range(shape_c[0] * shape_c[1]):
    #     #  c[i] = (i + 1) % 3
    #     c[i] = 1
    a.tofile(f'data/A.bin')
    b.tofile(f'data/B.bin')
    c.tofile(f'data/C.bin')
    alpha = np.random.uniform(-1, 1, 1).astype(np.float32)
    beta  = np.random.uniform(-1, 1, 1).astype(np.float32)
    # alpha = np.random.randint(-10, 10, size=1).astype(np.float32)
    # beta  = np.random.randint(-10, 10, size=1).astype(np.float32)
    # alpha = np.ones(1).astype(np.float32) * 1
    # beta  = np.ones(1).astype(np.float32) * 0
    alpha.tofile(f'data/alpha.bin')
    beta.tofile(f'data/beta.bin')

    a = a.reshape(shape_a[1], shape_a[0])
    a = a.T
    b = b.reshape(shape_b[1], shape_b[0])
    b = b.T
    c = c.reshape(shape_c[1], shape_c[0])
    c = c.T

    if (transA > 0):
        a_real = a[0:K,:]
    else:
        a_real = a[0:M,:]
    if (transB > 0):
        b_real = b[0:N,:]
    else:
        b_real = b[0:K,:]
    

    if (transA > 0):
        a_real = a_real.T
    if (transB > 0):
        b_real = b_real.T

    c_real = c[0:M,:]
    c_real = (beta.astype(np.float32) * c_real.astype(np.float32) + alpha.astype(np.float32) * np.matmul(a_real.astype(np.float32),b_real.astype(np.float32))).astype(np.float32)    
    # print(c)
    c[0:M,:] = c_real
    c = np.ascontiguousarray(c.T)

    c.tofile(f'data/C_expect.bin')

if __name__ == "__main__":
    M = 2
    N = 2
    K = 888
    lda = 1000
    ldb = 1000
    ldc = 1000

    transA = 0
    transB = 0

    parser = argparse.ArgumentParser()
    parser.add_argument('transA', action='store', type=int)
    parser.add_argument('transB', action='store', type=int)
    parser.add_argument('M', action='store', type=int)
    parser.add_argument('N', action='store', type=int)
    parser.add_argument('K', action='store', type=int)
    parser.add_argument('lda', action='store', type=int)
    parser.add_argument('ldb', action='store', type=int)
    parser.add_argument('ldc', action='store', type=int)

    args = parser.parse_args()

    transA = args.transA
    transB = args.transB
    M = args.M
    N = args.N
    K = args.K
    lda = args.lda
    ldb = args.ldb
    ldc = args.ldc

    calc_expect_func(transA, transB, M, N, K, lda, ldb, ldc)
