import argparse
import numpy as np
import torch
import torch_npu
import csv
import time

parser = argparse.ArgumentParser()
parser.add_argument('--trans', action='store', type=int, default=0)
parser.add_argument('-M', action='store', type=int, default=1024)
parser.add_argument('-N', action='store', type=int, default=1024)
args = parser.parse_args()
trans = args.trans
M = args.M
N = args.N

def performance(trans, M, N):

    A = torch.Tensor(np.random.uniform(0,1, size=(M, N)).astype(np.float16)).npu()
    x = torch.Tensor(np.random.uniform(0,1, size=(N, 1)).astype(np.float16)).npu()
    if trans:
        A = torch.Tensor(np.random.uniform(0,1, size=(M, N)).astype(np.float16)).npu()
        x = torch.Tensor(np.random.uniform(0,1, size=(1, M)).astype(np.float16)).npu()

    start_time = 0
    end_time = 0
    iteration = 1000

    if trans:
        torch.npu.synchronize()
        start_time = time.time()
        for _ in range(iteration):
            y = torch.matmul(x,A)
        torch.npu.synchronize()
        end_time = time.time()
    else:
        torch.npu.synchronize()
        start_time = time.time()
        for _ in range(iteration):
            y = torch.matmul(A,x)
        torch.npu.synchronize()
        end_time = time.time()

    average_duration = (end_time - start_time)/iteration #ms
    gflops = 2 * M * N / (average_duration) *1e-12
    print(trans," ",M," ", N," ", gflops) 
    with open('result.csv','a',newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([trans, M, N, gflops])
        
for trans in [1,0]:
    for M in range(1024,17408,1024):
        N = 17408 - M
        performance(trans, M, N)

for trans in [0,1]:
    for M in [1024, 2048, 4096, 8192, 16384, 32768]:
        performance(trans, M, M)