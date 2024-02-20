#!/bin/bash

OP_NAME="hgemv"
trans=$1
M=$2
N=$3
lda=$4
alpha=$5
beta=$6
incx=$7
incy=$8

isVerify=0 # 是否验证算子的正确性
if [[ ${9} == "prof" ]]; then # 执行性能测试不要需要验证正确性
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
if [[ ${9} != "prof" ]]; then
    python3 ${OP_NAME}.py --trans $trans -M $M -N $N --lda $lda --alpha $alpha --beta $beta --incx $incx --incy $incy
fi

make

if [[ ${9} == "prof" ]]; then
    rm -rf prof/*
    msprof --application="./build/${OP_NAME} $trans $M $N $lda $alpha $beta $incx $incy $isVerify" --output=./prof --aic-metric=L2Cache
    python3 prof.py `find ./prof -name "op_*.csv"` $trans $M $N $lda $alpha $beta $incx $incy
else
    ./build/${OP_NAME} $trans $M $N $lda $alpha $beta $incx $incy $isVerify
fi

# 删除不需要的数据和中间文件
# rm -rf data/*.bin prof/PROF*