# CUDA 基础速查

CUDA 核函数用 `<<<gridSize, blockSize>>>` 启动，例如 `kernel<<<4,256>>>();` 表示 4 个 Block、每个 Block 256 个 Thread，总线程数为 `4×256=1024`。

CUDA 线程层级可记为：

```text
Grid → Block → Warp → Thread
```

其中一次 Kernel 启动对应一个 Grid；Grid 包含多个 Block；Block 包含多个 Thread；GPU 实际以 Warp 为基本调度单位，**1 Warp = 32 Threads**。

常用内置变量：

```cpp
threadIdx   // 当前线程在 Block 中的位置
blockIdx    // 当前 Block 在 Grid 中的位置
blockDim    // 一个 Block 的尺寸
gridDim     // Grid 的尺寸
```

它们都有 `.x/.y/.z` 三个维度。一维情况下，全局线程编号最常用：

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

`uint3` 本质上保存 `x、y、z` 三个无符号整数；`dim3` 也有 `x、y、z`，但增加了构造函数、默认值和与 `uint3` 的转换能力，例如：

```cpp
dim3 block(16,16);   // 实际为 (16,16,1)
```

一个 Block 的线程总数在现代 NVIDIA GPU 上通常最多为 **1024**。Block Size 常用 `128/256/512`，因为它们都是 Warp Size 32 的整数倍，例如 256 Threads = 8 Warps。

GPU 中线程数通常远大于 CUDA Core 数，Block 数也通常要多于 SM 数量。原因是 GPU 需要大量可运行 Warp：当某个 Warp 等待显存时，可以切换到其他 Warp，从而隐藏延迟、提高硬件利用率。**Thread 并不是与 CUDA Core 一一对应。**

同一个 Warp 内如果线程执行不同分支，会产生 **Warp Divergence**，可能降低执行效率。

SM（Streaming Multiprocessor）是 GPU 的主要计算单元，Block 会被调度到 SM 上执行；SM 内包含 CUDA Core、Warp Scheduler、寄存器和 Shared Memory 等资源。

Kernel 启动通常是异步的：

```cpp
kernel<<<gridSize, blockSize>>>();
cudaDeviceSynchronize();
```

`cudaDeviceSynchronize()` 表示 CPU 等待此前提交给 GPU 的工作执行完成后再继续。

NVCC 编译设备代码的大致流程：

```text
CUDA C/C++ → PTX → CUBIN → GPU Machine Code
```

其中 `PTX` 是中间/虚拟指令，`CUBIN` 是针对具体 GPU 架构生成的机器码。

架构参数可记为：

```text
compute_XY → 虚拟架构 / PTX
sm_XY      → 实际 GPU 架构 / 机器码
```

例如：

```bash
-arch=compute_80 -code=sm_80
```

表示按 `compute_80` 生成 PTX，再生成 `sm_80` 机器码。

完整写法：

```bash
-gencode arch=compute_80,code=sm_80
```

`-gencode` 可以写多次，用于同时生成多个 GPU 架构版本。

如果写：

```bash
-gencode arch=compute_80,code=compute_80
```

则保留 PTX，程序运行时由 NVIDIA Driver 进行 **JIT（即时编译）**，把 PTX 编译成当前 GPU 的机器码。

最常见的简化写法：

```bash
nvcc test.cu -arch=sm_80
```

---

## 最后速记

```text
Kernel → Grid → Block → Warp → Thread

1 Warp = 32 Threads
1 Block 通常 ≤ 1024 Threads
常见 Block Size：128 / 256 / 512

idx = blockIdx.x * blockDim.x + threadIdx.x

Threads >> CUDA Cores
Blocks  >> SM 数量

cudaDeviceSynchronize()
= CPU 等 GPU 完成

.cu → PTX → CUBIN → GPU

compute_XY = PTX / 虚拟架构
sm_XY      = GPU / 机器码
```
