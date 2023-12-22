#!/bin/bash

set -e

CANN_DIR=${ASCEND_HOME_PATH}

mkdir -p input output

rm -rf *.o output/*.bin input/*.bin

python3 matmul.py

ccec -std=c++17 -c -x cce -O2 matmul.cpp --cce-aicore-arch=dav-c220-cube \
    -I${CANN_DIR}/compiler/tikcpp/tikcfw -I${CANN_DIR}/compiler/tikcpp/tikcfw/interface -I${CANN_DIR}/compiler/tikcpp/tikcfw/impl \
    -o matmul.o
# 编译不可链接的纯device.o: 加一个--cce-aicore-only参数

g++ -O2 -c main.cpp -fPIC -I${CANN_DIR}/include -o main.o

ccec -o test main.o matmul.o --cce-fatobj-link -L${CANN_DIR}/lib64 -lascendcl -lruntime -lstdc++

./test

echo "md5sum:";md5sum output/*.bin
