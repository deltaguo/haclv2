import csv
import sys

file_path = sys.argv[2]
M = int(sys.argv[3])
N = int(sys.argv[4])
K = int(sys.argv[5])
lda = int(sys.argv[6])
ldb = int(sys.argv[7])
ldc = int(sys.argv[8])
with open(file_path, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    # next(reader)
    time_us = 0.0
    for row in reader:
        # print(row['Task Type'])
        if (row['Task Type'] == "AI_CORE"):
            time_us += float(row['Task Duration(us)'])
Mflops = 2.0 * M * N * K * 1e-6
# 需要保留 time_us: 脚本会按" time_us: "获取时间
print("M: ", M, "N: ", N, "K: ", K, "lda: ", lda, "ldb: ", ldb, "ldc: ", ldc, "time_us: ", time_us, "Tflops: ", Mflops / time_us, "utilization ratio: ", Mflops / time_us / 61.44)