#!/bin/bash

OP_NAME="ppmatmul"

batch=$1
trans_a=$2
trans_b=$3
M=$4
N=$5
K=$6
lda=$7
ldb=$8
ldc=$9

isVerify=0 # 是否验证算子的正确性
if [[ ${10} == "prof" ]]; then # 执行性能测试不要需要验证正确性
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

python3 ${OP_NAME}.py $M $N $K $lda $ldb $ldc --batch $batch --trans_a $trans_a --trans_b $trans_b

make

if [[ ${10} == "prof" ]]; then
    rm -rf prof/*
    msprof --application="./build/${OP_NAME} $batch $trans_a $trans_b $M $N $K $lda $ldb $ldc $isVerify" --output=./prof --aic-metric=L2Cache
    python3 prof.py `find ./prof -name "op_*.csv"` $batch $M $N $K $lda $ldb $ldc
else
    ./build/${OP_NAME} $batch $trans_a $trans_b $M $N $K $lda $ldb $ldc $isVerify
fi

# 删除不需要的数据和中间文件
rm -rf data/*.bin prof/PROF*