- 2024.1.21 写完cube上hgemv功能
  - 性能最高0.76TFlops（910B4峰值带宽0.8T/s，roofline顶约为0.8T/s）
  - 存在原子加的精度问题
- 2024.1.29 完成splict方案，以float将结果保存在gm上
- 2024.2.20 完成带有alpha beta的mix算子

1. 执行gemv

   ```shell
   cd hgemv
   ./run.sh 0 8192 8192 8192 1.0 1.0 1 1
   ```

2. ascblasHgemm测试

   ```
   cd ascblas/tests/ascblasHgemm
   ./test.sh
   ```

3. mindspore测试(镜像：ascendhub.huawei.com/public-ascendhub/mindspore-modelzoo:23.0.0)

   ```shell
   #进入镜像后设置，环境变量
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   cd mindspore
   ./test.sh
   ```
   
若出现DOS风格换行符异常:
```
vim script.sh
:set ff=unix
:wq
```

