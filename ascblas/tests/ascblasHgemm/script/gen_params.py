import sys
import numpy as np
import csv
import os

def gen_test_data(kernel_name, times, src_type="int32"):
    if times < 0:
        print("times must be greater than or equal to 0!")
        sys.exit(1)
    if ("result" in os.getcwd()):
        test_dim_csv_filepath = "../result/" + kernel_name + "_data.csv"
    else:
        test_dim_csv_filepath = "result/" + kernel_name + "_data.csv"

    low_M = 1
    high_M = 10000
    low_N = 1
    high_N = 10000
    low_K = 1
    high_K = 10000

    with open(test_dim_csv_filepath, "w") as f_output:
        f_output.write("transA,transB,M,N,K,lda,ldb,ldc\n")
    i = 0
    while i < times:
        transA = np.random.randint(0, 2, dtype=np.int32)
        transB = np.random.randint(0, 2, dtype=np.int32)
        M = np.random.randint(low_M, high_M, dtype=np.int32)
        N = np.random.randint(low_N, high_N, dtype=np.int32)
        K = np.random.randint(low_K, high_K, dtype=np.int32)
        if transA == 0: #不转置的情况,lda必须大于等于M
            lda = np.random.randint(M, high_M, dtype=np.int32)
        else:  #转置的情况,lda必须大于等于K
            lda = np.random.randint(K, high_K, dtype=np.int32)
        if transB == 0: #不转置的情况,ldb必须大于等于K
            ldb = np.random.randint(K, high_K, dtype=np.int32)
        else:  #转置的情况,ldb必须大于等于N
            ldb = np.random.randint(N, high_N, dtype=np.int32)
        ldc = np.random.randint(M, high_M, dtype=np.int32)

        with open(test_dim_csv_filepath, "a") as f_output:
            writer = csv.writer(f_output)
            # 写入一行数据
            row = [transA, transB, M, N, K, lda, ldb, ldc]
            writer.writerow(row)
        i += 1

if __name__ == "__main__":
    kernel_name = sys.argv[1]
    times = int(sys.argv[2])
    gen_test_data(kernel_name, times)