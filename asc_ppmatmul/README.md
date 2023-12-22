## 文件结构
``` shell
asc_ppmatmul
|-- build
|-- CAmodel
|-- data
|-- data_utils.h
|-- main.cpp
|-- ppmatmul.cpp
|-- ppmatmul.py
|-- run.sh
|-- README.md
```

- build 可执行文件存放目录
- CAmodel CAmodel 测试目录
- data 点bin文件的存放目录
- data_utils.h 辅助函数的头文件
- main.cpp 测试主函数文件
- ppmatmul.cpp 内核函数文件
- ppmatmul.py 数据生成脚本
- run.sh 测试脚本
- README.md

# 运行步骤
1. 加载环境变量
``` shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```
2. 编译文件
``` shell
make
```
- CAmodel 需要额外运行
``` shell
make ca
```
3. 运行算子测试算子正确性
- ./run.sh [batch] [M] [N] [K] [trans_a] [trans_b] [prof]
- prof 表示是否测试性能，不加只看结果是否正确
``` shell
./run.sh 1 0 0 1024 1024 1024 1024 1024 1024

```
4. 运行算子测试算子性能
``` shell
./run.sh 1 0 0 1024 1024 1024 1024 1024 1024 prof

./run.sh 1 0 0 2048 2048 2048 2048 2048 2048 prof

./run.sh 1 0 0 128 128 128 128 128 128 prof

./run.sh 1 0 0 256 256 256 256 256 256 prof

./run.sh 1 0 0 8192 8192 8192 8192 8192 8192

./run.sh 1 0 0 7777 7777 7777 8192 8192 8192

./run.sh 1 0 0 17 17 17 32 32 32

./run.sh 1 0 0 64 64 64 128 128 128

```
5. 运行CAmodel测试
``` shell
cd CAmodel

./run_sim.sh
```
