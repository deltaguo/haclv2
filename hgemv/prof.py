import csv
import sys

# print(sys.argv[1])
# print(sys.argv[2])
# print(sys.argv[3])
# print(sys.argv[4])
# print(sys.argv[5])
# print(sys.argv[6])

file_path = sys.argv[2]
trans = int(sys.argv[3])
M = int(sys.argv[4])
N = int(sys.argv[5])
lda = int(sys.argv[6])
incx = int(sys.argv[7])
incy = int(sys.argv[8])

with open(file_path, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    # next(reader)
    time_us = 0.0
    for row in reader:
        time_us += float(row['Task Duration(us)'])
    print(time_us)
Mflops = 2.0 * M * N * 1e-6
print("trans: ", trans, "M: ", M, "N: ", N, "lda: ", lda, "incx: ", incx, "incy: ", incy, "Tflops: ", Mflops / time_us)