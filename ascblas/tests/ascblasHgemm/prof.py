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
    time_us_list = [float(row['Task Duration(us)']) for row in reader if row['Task Type'] == "AI_CORE"]
    time_us = sum(time_us_list[1:]) / len(time_us_list[1:])  # 忽略第一次的任务持续时间，取平均值

Mflops = 2.0 * M * N * K * 1e-6
print("M: ", M, "N: ", N, "K: ", K, "lda: ", lda, "ldb: ", ldb, "ldc: ", ldc, "time_us: ", time_us, "Tflops: ", Mflops / time_us, "utilization ratio: ", Mflops / time_us / 245.76)
with open('../result.csv','a',newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([M, N, K, Mflops / time_us])