- 2024.1.21 写完cube上hgemv功能
  - 性能最高0.76TFlops（910B4峰值带宽0.8T/s，roofline顶约为0.8T/s）
  - 存在原子加的精度问题
- 2024.1.29 完成splict方案，以float将结果保存在gm上
- 接下来
  1. tiling方案，优化访存
  2. 使用vector单元进行前处理后处理
  4. 核间同步

昇腾910B上的一些要点:

## 转置处理：

在cube上，ABC矩阵的分形格式各不相同，A(zZ)，B(nZ)，C(zN)，无论hgemv是否对矩阵转置，得到的都是一个向量y，为了让向量在L0C上尽可能连续存放(zN下应该是一个行向量)，在所有情况下都应该用L0A来存放向量x，L0B来存放矩阵A。

1. Trans = 0(不转置)：

   此时计算的是矩阵A右乘列向量x得到列向量y。由于列向量x以行向量形式存在L0A中，所以矩阵A需要进行转置，该过程在L12L0进行。最后得到的行向量与理应得到的列向量等价。

2. Trans=1(转置)

   此时计算的是矩阵A的转置右乘列向量x得到列向量y。将 $A^T*x=y$ 等号两边取转置得到 $(x*A)^T=y^T$，因此可以直接将向量x搬运到L0A，矩阵A搬运到L0B。最后得到的行向量也与列向量等价。

## 计算过程

计算流程是对矩阵A在L1上进行分块，然后基于L1上的块，在L0上继续分块。L1上的块尺寸为(M1, N1)，L0上块的尺寸为(M0，N0)。串行的流程为每个AI core处理累加方向上所有的L1块，GM搬运数据到L1，L1中的数据分一次或多次搬运到L0AB上，用cube完成计算，写到L0C，计算完本次L1上的所有数据后将L0C的结果写回GM。

## 双缓冲（Double buffer）

L1，L0AB，L0C上均使用double buffer（L0AB上影响较小，因为性能很大程度取决于MTE2的速度，MTE1对性能的影响只有流水线末尾那一小段）。使用了double buffer之后可以发现MTE2的所有操作在timeline示意图中几乎连续，并且覆盖住了MTE1，CUBE，FIX操作。所以此时**提升空间在于如何优化MTE2的访存**。

## 增大并行度（splict k）

MTE2的访存指令有个特点，在访问矩阵的时候第一维的大小越连续带宽利用率越高，另外gemv计算对数据复用的要求低，因此在列优先情况下，基块尺寸如果遵循M1>>N1，可以达到最大的读取性能，当然M1和N1的尺寸也受到了存储单元的影响。

但M1>>N1带来的问题是，极有可能使得划分出的L1的块数量不够（例如基块大小为（1024，64）时，计算规模为（10240，10240）也只划分了10个block，910B4上有20个AI core，其他型号更多。导致NPU上的AIcore利用率低，并行度低。因为目前只在非累加纬度进行了分块，所以为了增大并行度，需要在累加维度继续分块。

此时不同的AI core计算相同位置上的部份结果，使用原子操作累加在GM上。虽然GM上原子加很耗时，但也会被下一轮的读取大矩阵的MTE2盖住。所以对总体性能影响不大，好处是增加了并行度。

**目前带来的问题是将L0C的fp32结果原子加到GM上的fp16上有精度问题，正在找解决方案。（已解决）**

## Tiling方案(施工中)

根据910B4的存储空间大小，(M1,N1)的上限为双缓冲下的(1024, 64)，单缓冲下的(2048, 64)，（M0, N0）上限为双缓冲下的(512, 32)，单缓冲下的(1024, 32)。

M1，N1和M0，N0受到以下硬性限制（为确保计算流程不出错）：

1. M1，N1需要能够整除M0，N0，且商为偶数（为了双缓冲）。
2. 双缓冲情况下，对于单个基块而言，如果L0在L1上迭代读取次数>1（双缓冲），那么连续两次迭代的cube计算不能写到同一位置，否则会发生数据踩踏。所以针对该情况，非转置时M0<M1，转置时N0<N1。

为提高计算效率，M1，N1，M0，N0的选择需要使得最终分出来块的数量向上接近AI core数量的整数倍，与此同时申请的逻辑AIcore数量要越少越好。