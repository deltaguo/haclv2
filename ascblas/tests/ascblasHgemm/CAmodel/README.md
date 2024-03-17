# CAmodel 测试，用于得到内核的流水线。进行下一步优化

## 如何执行整个过程
``` shell

make ..

make ca

cd CAmodel

./run_sim.sh

export PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/bin:$PATH

msopgen sim -c core0 -d ./model -out wave -mix
```
进行wave文件夹中的json用chrome::tracing打开，即可得到流水线

## 编写自己算子的CAmodel
- 需要的文件
  1. cce 内核文件
  2. json 内核配置文件
  3. python调用文件
- 注意的点，编写内核时按照示例编写，CAmodel模式的内核，指针顺序是有顺序的。先输入数据指针，再输出数据指针。其次是tiling指针。