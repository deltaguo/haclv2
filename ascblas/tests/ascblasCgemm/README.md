## 文件结构
``` shell
ascblasCgemm
|-- build
|-- CAmodel
|-- data
|-- prof
|-- result
|-- script
|-- ascblasCgemm_gen_data.py
|-- ascblasCgemm.cce
|-- ascblasCgemm.h
|-- main.cpp
|-- makefile
|-- prof.py
|-- README.md
|-- run.sh
```

- build 可执行文件存放目录
- CAmodel CAmodel 测试目录
- data 点bin文件的存放目录
- prof 保存PROF的临时文件
- result 保存批量测试的结果
- script 批量测试的脚本
- ascblasCgemm_gen_data.py 验证正确性的数据生成
- ascblasCgemm.cce 核函数文件
- ascblasCgemm.h 核函数封装的头文件
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
- ./run.sh [M] [N] [K] [trans_a] [trans_b] [prof]
- prof 表示是否测试性能，不加只看结果是否正确
``` shell
./run.sh 1 1 20812 25257 315 22296 28414 27059

```
4. 运行算子测试算子性能
``` shell
./run.sh 0 0 3333 7777 4444 3333 4444 7777 prof

./run.sh 0 1 8192 8192 4096 8192 32768 8192 prof
./run.sh 0 1 8192 8192 8192 8192 32768 8192 prof
./run.sh 0 1 8192 8192 32768 8192 32768 8192 prof
./run.sh 0 1 8192 8192 65536 8192 65536 8192 prof

./run.sh 1 0 8192 8192 4096 4096 32768 8192 prof
./run.sh 1 0 8192 8192 8192 8192 32768 8192 prof
./run.sh 1 0 8192 8192 9216 9216 32768 8192 prof
./run.sh 1 0 8192 8192 10240 10240 32768 8192 prof
./run.sh 1 0 8192 8192 12288 12288 32768 8192 prof
./run.sh 1 0 8192 8192 32768 32768 32768 8192 prof


./run.sh 0 1 8192 8192 4096 8192 32768 8192 prof
./run.sh 0 1 4096 4096 4096 4096 4096 4096 prof
./run.sh 0 1 4096 4096 32768 4096 32768 4096 prof
./run.sh 0 1 256 256 128 256 256 256 prof
./run.sh 0 1 128 128 128 128 128 128 prof
./run.sh 0 1 128 128 128 256 256 256 prof
./run.sh 0 1 256 256 32768 256 32768 256 prof
./run.sh 0 1 512 512 4096 512 4096 512 prof
./run.sh 0 1 512 512 32768 512 32768 512 prof
```
5. 运行CAmodel测试
``` shell
make ca

cd CAmodel

./run_sim.sh

export PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/bin:$PATH

msopgen sim -c core0 -d ./model -out wave -mix
```

6. 批量测试
- 测试性能
``` shell
python3 script/batch_prof.py
```
- 测试误差
``` shell
python3 script/batch_error.py
```