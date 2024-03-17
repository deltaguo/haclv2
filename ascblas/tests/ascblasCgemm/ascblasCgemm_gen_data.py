import numpy as np
import argparse

def gen_random_complex64_numpy(l, r, shape):
    return (np.random.uniform(l, r, size=shape)+1j*np.random.uniform(l, r, size=shape)).astype(np.complex64)
    # return (np.random.randint(l, r, size=shape)+1j*np.random.randint(l, r, size=shape)).astype(np.complex64)
    # if (isinstance(shape, tuple)):
    #     return (np.arange(shape[0]*shape[1])+1j*np.arange(1,shape[0]*shape[1]+1)).reshape(shape).astype(np.complex64)
    # else:
    #     return np.arange(shape).astype(np.complex64)

def process_x_valid(x, trans, dim):
    x_valid = x[:dim].T if trans else x[:dim]
    if trans == 2:
        x_valid = np.conj(x_valid)
    return x_valid

def gen_golden_data(transA, transB, M, N, K, lda, ldb, ldc):
    l, r = 0, 1
    # l, r = -10, 10
    shape_a = (lda, M) if transA else (lda, K)
    shape_b = (ldb, K) if transB else (ldb, N)
    shape_c = (ldc, N)

    a = gen_random_complex64_numpy(l, r, shape_a)
    b = gen_random_complex64_numpy(l, r, shape_b)
    c = gen_random_complex64_numpy(l, r, shape_c)
    alpha = gen_random_complex64_numpy(l, r, 1)
    beta = gen_random_complex64_numpy(l, r, 1)
    # alpha = np.complex64(2+3j)
    # beta = np.complex64(4+5j)

    # print(a, b, c, alpha, beta, sep='\n')
    for var, name in zip([a, b, c, alpha, beta], ['A', 'B', 'C', 'alpha', 'beta']):
        var.T.tofile(f'data/{name}.bin')

    a_vaild = process_x_valid(a, transA, K if transA else M)
    b_vaild = process_x_valid(b, transB, N if transB else K)

    # print(a_vaild.shape, b_vaild.shape)
    c[:M] = alpha * a_vaild @ b_vaild + beta * c[:M]
    # c[:M] = beta * c[:M]
    # c[:M] = a_vaild @ b_vaild
    # c[:M] = a_vaild @ b_vaild + beta * c[:M]

    # print(c)
    c.T.tofile(f'data/C_expect.bin')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='gen_golden_data')
    parser.add_argument('transA', type=int, default=0, choices=range(3))
    parser.add_argument('transB', type=int, default=0, choices=range(3))
    parser.add_argument('M', type=int, default=256)
    parser.add_argument('N', type=int, default=256)
    parser.add_argument('K', type=int, default=128)
    parser.add_argument('lda', type=int, default=256)
    parser.add_argument('ldb', type=int, default=128)
    parser.add_argument('ldc', type=int, default=256)

    args = parser.parse_args()

    gen_golden_data(args.transA, args.transB, args.M, args.N, args.K, args.lda, args.ldb, args.ldc)
