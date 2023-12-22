import csv
import sys

# print(sys.argv[1])
# print(sys.argv[2])
# print(sys.argv[3])
# print(sys.argv[4])
# print(sys.argv[5])
# print(sys.argv[6])

file_path = sys.argv[2]
batch = int(sys.argv[3])
M = int(sys.argv[4])
N = int(sys.argv[5])
K = int(sys.argv[6])
lda = int(sys.argv[7])
ldb = int(sys.argv[8])
ldc = int(sys.argv[9])
with open(file_path, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    # next(reader)
    time_us = 0.0
    for row in reader:
        time_us += float(row['Task Duration(us)'])
Mflops = 2.0 * batch * M * N * K * 1e-6
print("batch: ", batch, "M: ", M, "N: ", N, "K: ", K, "lda: ", lda, "ldb: ", ldb, "ldc: ", ldc, "Tflops: ", Mflops / time_us)