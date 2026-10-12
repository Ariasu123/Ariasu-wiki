# CUDA 程序调试与性能优化

## CUDA 错误检查与同步

**CUDA API 错误码**：`cudaMalloc`、`cudaMemcpy`、`cudaFree` 等运行时 API 返回 `cudaError_t`；`cudaSuccess` 表示成功，`cudaGetErrorString(err)` 将错误码转换为可读信息。`CHECK` 是自定义宏，并非 CUDA 内置函数。

```cpp
// error.cuh
#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CHECK(call)                                           \
do {                                                          \
    cudaError_t err = (call);                                 \
    if (err != cudaSuccess) {                                 \
        std::fprintf(stderr, "CUDA Error: %s\n",               \
                     cudaGetErrorString(err));                 \
        std::fprintf(stderr, "%s:%d\n", __FILE__, __LINE__);   \
        std::exit(EXIT_FAILURE);                              \
    }                                                         \
} while (0)
```

`#define` 在预处理阶段展开，行尾 `\` 用于续行；`do {...} while(0)` 将多语句宏包装成一条语句；`__FILE__` / `__LINE__` 给出宏调用位置。

**Kernel 错误检查**：Kernel 为 `__global__ void`，不能直接写 `CHECK(kernel<<<...>>>())`；启动通常异步，因此区分启动状态和执行完成后的错误。

```cpp
CHECK(cudaMalloc((void**)&d_x, M));
CHECK(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));

kernel<<<grid, block>>>(d_x, N);
CHECK(cudaGetLastError());       // 读取并清除最近错误状态，不等待 GPU
CHECK(cudaDeviceSynchronize()); // CPU 等待当前设备先前任务完成，并检查执行错误

CHECK(cudaMemcpy(h_x, d_x, M, cudaMemcpyDeviceToHost));
CHECK(cudaFree(d_x));
```

`cudaGetLastError()` 常用于发现 Kernel 启动配置错误，如 Block 线程数超限；`cudaPeekAtLastError()` 读取错误状态但不清除。`cudaDeviceSynchronize()` 可暴露非法显存访问等执行期错误，但同步的是当前设备先前提交的工作，错误可能来自更早的异步任务。

**同步开销**：常规默认流中的阻塞式 GPU → CPU `cudaMemcpy` 已会等待相关工作完成，通常无需额外 `cudaDeviceSynchronize()`；不是所有传输方向、流和 API 都会同步。调试时可显式同步定位问题，性能敏感循环中应避免无条件同步。

**Event 查询例外**：`cudaEventQuery(event)` 不等待事件完成；`cudaErrorNotReady` 表示事件尚未完成，并不一定是错误，不能直接用“非 `cudaSuccess` 即退出”的 `CHECK` 处理。

```bash
CUDA_LAUNCH_BLOCKING=1 ./app  # Linux：调试时临时让 Kernel 启动同步化
```

---

## GPU 内存与并发错误检测

教材中的 `cuda-memcheck` 已被新版 CUDA 的 **`compute-sanitizer`** 替代（CUDA 12.0 起移除旧工具）。它通过运行时检查，辅助定位比普通 API 返回码更具体的 GPU 错误。

| 检测工具 | 主要检查内容 |
|---|---|
| `memcheck` | 非法、越界或非对齐的 GPU 内存访问等 |
| `racecheck` | Shared Memory 数据竞争 |
| `initcheck` | 未初始化的设备 Global Memory 读取 |
| `synccheck` | 同步原语的错误使用 |

```bash
nvcc -lineinfo app.cu -o app
compute-sanitizer --tool memcheck ./app
compute-sanitizer --tool racecheck ./app
compute-sanitizer --tool initcheck ./app
compute-sanitizer --tool synccheck ./app
```

常见越界：`if (n > N) return;` 漏掉 `n == N`，应写成 `if (n >= N) return;`。`-lineinfo` 保留设备代码行号信息；检测会降低运行速度，`0 errors` 不代表程序绝对正确。

---

## CUDA Event 计时

使用两个 CUDA Event 记录同一 Stream 中的起止位置，计算 GPU 时间间隔，单位为 **ms**。测量 Kernel 耗时时应只包围需要计时的操作，并等待结束事件完成。

```cpp
cudaEvent_t start, stop;
CHECK(cudaEventCreate(&start));
CHECK(cudaEventCreate(&stop));

CHECK(cudaEventRecord(start));
kernel<<<grid, block>>>();
CHECK(cudaGetLastError());
CHECK(cudaEventRecord(stop));
CHECK(cudaEventSynchronize(stop));

float elapsed_ms = 0;
CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
CHECK(cudaEventDestroy(start));
CHECK(cudaEventDestroy(stop));
```

`&start` / `&stop` 用于写入新创建的 Event 句柄；`start` / `stop` 传入已有句柄；`&elapsed_ms` 用于让函数写入计时结果。旧版教材在 WDDM 场景中额外调用 `cudaEventQuery(start)` 促使队列提交，并非现代程序的固定必要步骤。性能测试应预热、多次运行，避免只取首次结果。

---

## 数据传输、显存带宽与算术强度

**区分两类数据传输**：`cudaMemcpy` 在 CPU 内存与 GPU 显存之间复制数据（独立 GPU 通常经 PCIe）；Kernel 的有效显存带宽指 GPU 计算期间**设备显存的有效读写量 ÷ Kernel 时间**，不是 CPU↔GPU 的传输速度。

```cpp
z[n] = x[n] + y[n]; // 读取 x、y，写入 z；一次浮点加法
```

单精度 `float`（4 B）每元素访问约 `3 × 4 = 12 B`；若 `N = 10^8`、Kernel 耗时约 `3.3 ms`：

\[
B_{\mathrm{eff}}=\frac{3\times N\times4\ \mathrm{B}}{t}\approx360\ \mathrm{GB/s}
\]

教材的 RTX 2070 理论显存带宽约 `448 GB/s`，所以数组加法接近显存带宽瓶颈，属于 **Memory-bound（访存主导）**。教材示例中仅 Kernel 约 `3.3 ms`，计入 CPU↔GPU 拷贝后约 `180 ms`，说明 **Kernel 快不等于端到端程序快**。

**算术强度（Arithmetic Intensity）**：单位内存访问量对应的浮点计算量，常以 `FLOPs/Byte` 表示。算术强度低往往受带宽限制；算术强度高时更可能受计算吞吐限制，即 **Compute-bound（计算主导）**。

```cpp
real x_tmp = d_x[n];
while (sqrt(x_tmp) < x0) {
    ++x_tmp;
}
d_x[n] = x_tmp;
```

这里每个线程读入一次、循环中反复计算、最后写回一次；`x_tmp` 通常保存在寄存器中，减少全局显存访问。GPU 版本以全局线程索引代替 CPU 外层 `for`，每个线程内部仍可有 `while`。

**FMA**：`d = a*b + c` 是一条融合乘加指令，1 次乘法 + 1 次加法，按 **2 FLOPs** 计。书中由 RTX 2070 的 `6.5 TFLOPS` 推导的 `52 TB/s` 是理想化的**寄存器操作数访问需求速率**，不是显存的实测带宽。

---

## 并行规模与 GPU 加速条件

GPU 由多个 **SM** 构成，SM 包含 CUDA Core、Warp Scheduler、寄存器和 Shared Memory 等；Kernel 的线程组织为 `Grid → Block → Warp（32 Threads）→ Thread`。Block 分配到 SM 执行，**Thread 与 CUDA Core 不一一对应**。

线程太少时 GPU 资源空闲，启动开销占比高；线程增加可提高利用率，规模足够大后加速比逐渐趋于饱和。获得较高 GPU 加速的三个关键条件（必要但不充分）：**减少 Host↔Device 传输比例、提高算术强度、提供足够并行线程**。

---

## 编译选项与数学函数

```cpp
#ifdef USE_DP
    typedef double real; // FP64，一般 8 B
#else
    typedef float real;  // FP32，一般 4 B
#endif
```

```bash
nvcc -O3 app.cu -o app           # 较高主机编译优化等级；此示例默认 float
nvcc -O3 -DUSE_DP app.cu -o app  # 定义 USE_DP，选用 double
```

`-D` 定义预处理宏，`#ifdef` 按宏是否存在保留对应代码；`-O3` 在 NVCC 中主要控制主机编译器的优化级别。`real` 是自定义类型别名，不是 C++ 标准类型。

CUDA 提供单精度和双精度的数学函数（平方根、三角函数、指数、对数等）。`sqrt(float)` / `sqrt(double)` 通过重载区分类型，`sqrtf(float)` 明确指定单精度；部分 intrinsics 提供特定舍入方式或性能特征，按精度需求选择，不能一概认为“内建函数都更快”。

**新版性能分析工具**：教材使用的 `nvprof` 属于旧工作流；现代 CUDA 通常使用 `nsys`（Nsight Systems，查看 CPU/GPU 时间线、Memcpy）和 `ncu`（Nsight Compute，分析单个 Kernel 的瓶颈）。

```bash
nsys profile -o report ./app
ncu ./app
```

---

## 速记

```text
正确性：
CHECK(cudaMalloc(...))      检查 API 返回码
cudaGetLastError()          读取并清除最近错误，不等待 GPU
cudaDeviceSynchronize()     CPU 等当前 GPU 先前任务完成
cudaEventQuery()            非阻塞事件查询，NotReady 并非失败
compute-sanitizer          memcheck / racecheck / initcheck / synccheck

性能：
cudaEventElapsedTime()     Event 时间差，单位 ms
-O3 / -DUSE_DP              编译优化 / 开启 double 分支
Memory-bound               显存带宽限制
Compute-bound              计算吞吐限制
Arithmetic Intensity       FLOPs / Byte
GPU 加速核心                少数据往返、多数据复用、足够多线程
nsys / ncu                  整体时间线 / Kernel 性能分析
```

参考：樊哲勇《CUDA 编程：基础与实践》错误检测与 GPU 加速相关内容；[NVIDIA Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html) · [NVIDIA Nsight Systems](https://docs.nvidia.com/nsight-systems/) · [NVIDIA Nsight Compute](https://docs.nvidia.com/nsight-compute/)
