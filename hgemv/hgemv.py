import numpy as np
import argparse
import os

def generate_test_data(trans, M, N, lda, incx, incy):
    #np.random.seed(42)
    left = -1
    right = 1
    os.makedirs('data', exist_ok=True)
    mA = np.random.uniform(left,right, size=(lda, N)).astype(np.float16)
    #mA = np.random.randint(left,right, size=(lda, N)).astype(np.float16)
    #mA = np.arange(0.01,(M*N)*0.01+0.01,0.01).reshape((M,N)).astype(np.float16)
    mA.ravel('F').tofile('data/matrixA.bin')
    mA_real = mA[0:M,0:N]
    vX = None
    vY = None
    if trans == 0:
        vX = np.random.uniform(left,right, size=(N * incx, 1)).astype(np.float16)
        #vX = np.random.randint(left,right, size=(N * incx, 1)).astype(np.float16)
        #vX = np.arange(0.01,N*0.01+0.01,0.01).reshape((N,1)).astype(np.float16)
        #vY = np.random.uniform(left,right, size=(M * incy, 1)).astype(np.float16)
        vY = np.random.uniform(0,0, size=(M * incy, 1)).astype(np.float16)
    else:
        vX = np.random.uniform(left,right, size=(1, M * incx)).astype(np.float16) 
        #vX = np.random.randint(left,right, size=(1, M * incx)).astype(np.float16) 
        vY = np.random.uniform(0,0, size=(1, N * incy)).astype(np.float16)

    vX.ravel('F').tofile('data/vectorX.bin') 
    vY.ravel('F').tofile('data/vectorY.bin') 
    # print(mA)
    # print(vX)
    # print(vY)
    if trans == 0:
        vX_real = vX[0:N*incx:incx,:]
        vY_real = vY[0:M*incy:incy,:]
        vY_real = (vY_real.astype(np.float32) + np.matmul(mA_real.astype(np.float32), vX_real.astype(np.float32))).astype(np.float16)
        vY[0:M*incy:incy,:] = vY_real
    else:
        vX_real = vX[:,0:M*incx:incx]
        vY_real = vY[:,0:N*incy:incy]
        vY_real = (vY_real.astype(np.float32) + np.matmul(vX_real.astype(np.float32), mA_real.astype(np.float32))).astype(np.float16)
        vY[:,0:N*incy:incy] = vY_real
    vY.ravel('F').tofile('data/vectorR.bin')
    # print(vY)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trans', action='store', type=int, default=0)
    parser.add_argument('-M', action='store', type=int, default=1024)
    parser.add_argument('-N', action='store', type=int, default=1024)
    parser.add_argument('--lda', action='store', type=int, default=1024)
    parser.add_argument('--incx', action='store', type=int, default=1)
    parser.add_argument('--incy', action='store', type=int, default=1)

    args = parser.parse_args()
    # print(args.trans, args.M, args.N, args.lda, args.incx, args.incy)
    generate_test_data(args.trans, args.M, args.N, args.lda, args.incx, args.incy)