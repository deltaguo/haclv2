#!/bin/bash

OP_NAME="hgemv"
trans=$1
M=$2
N=$3
lda=$4
incx=$5
incy=$6

isVerify=0 # 是否验证算子的正确性
if [[ ${7} == "prof" ]]; then # 执行性能测试不要需要验证正确性
    isVerify=0
else
    isVerify=1
fi

# 删除之前不需要的文件
mkdir -p data build

# rm -rf  build/*.o

# 载入环境变量
# source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -e
CANN_DIR=${ASCEND_HOME_PATH}
python3 ${OP_NAME}.py --trans $trans -M $M -N $N --lda $lda --incx $incx --incy $incy

make

if [[ ${7} == "prof" ]]; then
    rm -rf prof/*
    msprof --application="./build/${OP_NAME} $trans $M $N $lda $incx $incy $isVerify" --output=./prof --aic-metric=L2Cache
    #python3 prof.py `find ./prof -name "op_*.csv"` $batch $M $N $K $lda $ldb $ldc
else
    ./build/${OP_NAME} $trans $M $N $lda $incx $incy $isVerify
fi

# 删除不需要的数据和中间文件
#rm -rf data/*.bin prof/PROF*