import argparse
import csv
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--filepath', action='store', type=str, default="")
parser.add_argument('--trans', action='store', type=int, default=0)
parser.add_argument('-M', action='store', type=int, default=1024)
parser.add_argument('-N', action='store', type=int, default=1024)

args = parser.parse_args()
filepath = args.filepath
trans = args.trans
M = args.M
N = args.N

with open(filepath, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    time_ms = 0.0
    for row in reader:
        time_ms += float(row['total_time'])
    print(time_ms)
Gflops = 2.0 * M * N * 1e-9
print("trans: ", trans, "M: ", M, "N: ", N, "Tflops: ", Gflops / time_ms)
with open('result.csv','a',newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([trans, M, N, Gflops / time_ms])