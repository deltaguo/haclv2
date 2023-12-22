import numpy as np


def gen_golden_data():
    gm_dtype = np.float16
    M, K, N = 128, 128, 128

    x1_gm = np.random.randint(1, 10, (M, K)).astype(gm_dtype)
    x2_gm = np.random.randint(1, 10, (K, N)).astype(gm_dtype)
    golden = np.matmul(x1_gm.astype(np.float32), x2_gm.astype(np.float32)).astype(np.float32)

    x1_gm.tofile('./input/x1_gm.bin')
    x2_gm.tofile('./input/x2_gm.bin')
    golden.tofile('./output/golden.bin')


if __name__ == '__main__':
    gen_golden_data()
