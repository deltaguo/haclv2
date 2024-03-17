#!/bin/bash

OP_NAME="ascblasCgemm"

transA=$1
transB=$2
M=$3
N=$4
K=$5
lda=$6
ldb=$7
ldc=$8

verifyLevel=0 # 是否验证算子的正确性
if [[ ${9} == "prof" ]]; then # 执行性能测试不要需要验证正确性
    verifyLevel=0
elif [[ ${9} == "error" ]]; then # 执行功能测试输出误差
    verifyLevel=2
else
    verifyLevel=1
fi

# 删除之前不需要的文件
mkdir -p data

# rm -rf  build/*.o

# 载入环境变量
# source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -e
CANN_DIR=${ASCEND_HOME_PATH}

if [[ ${9} != "prof" ]]; then
    python3 ${OP_NAME}_gen_data.py $transA $transB $M $N $K $lda $ldb $ldc
fi

if [[ ${verifyLevel} == 1 ]]; then
make > /dev/null
fi

cd build

if [[ ${9} == "prof" ]]; then
    rm -rf ../prof/*
    # msprof --application="./${OP_NAME} $transA $transB $M $N $K $lda $ldb $ldc $verifyLevel" --output=../prof > /dev/null
    msprof --application="./${OP_NAME} $transA $transB $M $N $K $lda $ldb $ldc $verifyLevel" --output=../prof
    python3 ../prof.py `find ../prof -name "op_*.csv"` $M $N $K $lda $ldb $ldc
else
    ./${OP_NAME} $transA $transB $M $N $K $lda $ldb $ldc $verifyLevel
fi

cd ../


# 删除不需要的数据和中间文件
# rm -rf data/*.bin prof/PROF*