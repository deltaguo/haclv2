import argparse
import numpy as np
import mindspore as ms
import mindspore.ops as ops
from mindspore import Profiler
import time
#pynative_synchronize=True
ms.set_context(device_target='Ascend', device_id=0)

def hgemv(A,x):
    y=ops.matmul(A,x)


def hgevm(A,x):
    y=ops.matmul(x,A)


def profile(func,A,x):
    for i in range(10):
        if i == 9:
            profiler.start()
        func(A,x)
        if i == 9:
            profiler.stop()
    
parser = argparse.ArgumentParser()
parser.add_argument('--trans', action='store', type=int, default=0)
parser.add_argument('-M', action='store', type=int, default=1024)
parser.add_argument('-N', action='store', type=int, default=1024)
args = parser.parse_args()
trans = args.trans
M = args.M
N = args.N
ms.set_context(device_target="Ascend")

profiler = Profiler(start_profile=False, output_path='./prof')

A = ms.Tensor(np.random.uniform(0,1, size=(M, N)).astype(np.float16)).astype(np.float16)
x = ms.Tensor(np.random.uniform(0,1, size=(N, 1)).astype(np.float16)).astype(np.float16)
if args.trans:
    A = ms.Tensor(np.random.uniform(0,1, size=(M, N)).astype(np.float16)).astype(np.float16)
    x = ms.Tensor(np.random.uniform(0,1, size=(1, M)).astype(np.float16)).astype(np.float16)

# profiler
if args.trans:
    profile(hgevm,A,x)
else:
    profile(hgemv,A,x)
profiler.analyse()
# start_time = time.time()
# for _ in range(1000):
#     y = ops.matmul(A,x)
# end_time = time.time()

# duration = (end_time - start_time)*1e-3 #ms
# gflops = 2 * M * N / (duration * 1e-3) *1e-9
# print(gflops)