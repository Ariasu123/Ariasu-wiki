# CUDA 程序的基本框架

## 1. CUDA 程序基本框架

```cpp
// 1. 头文件包含
#include <stdio.h>

// 2. 常量定义（或宏定义）
const int N = 10000;

// 3. C++ 自定义函数与 CUDA 核函数的声明（原型）
__global__ void add(const double *x, const double *y, double *z, int N);
void check(const double *z, int N);

// 4. 主函数：按顺序完成一次 CPU + GPU 计算
int main()
{
    // 分配主机与设备内存
    // 初始化主机中的数据
    // 将数据从主机复制到设备（Host → Device）
    // 调用核函数在设备中进行计算
    // 将数据从设备复制回主机（Device → Host）
    // 检查计算结果（按需）
    // 释放主机与设备内存
    return 0;
}

// 5. C++ 自定义函数与 CUDA 核函数的定义（实现）
// __global__ void add(...) { ... }
// void check(...) { ... }
```

- **声明**放在 `main()` 之前，方便主函数调用；**定义**可以放在 `main()` 之后。
- `h_` 通常表示 Host（CPU）端变量，`d_` 表示 Device（GPU）端变量；这是命名约定，不是特殊语法。
- CUDA 运行时一般会在首次需要设备的 API 调用时自动初始化设备，无须在这里显式初始化。


## 2. Host / Device 内存分配

```cpp
const int N = 10000;
const size_t M = N * sizeof(double); // 字节数，不是元素数

double *h_x = (double*)malloc(M);   // CPU 内存
double *d_x;
cudaMalloc((void**)&d_x, M);        // GPU 显存

free(h_x);
cudaFree(d_x);
```

- `malloc(M)` 申请 **M 字节**主机内存，返回 `void*`；也可用 `new double[N]`，但需对应 `delete[]`。
- `cudaMalloc((void**)&d_x, M)` 申请 M 字节设备内存，并把起始地址写入 `d_x`。
- `d_x` 是 `double*`，`&d_x` 是 `double**`；`(void**)` 将二级指针转换为 API 要求的类型。传 `&d_x` 是为了让函数**修改指针 d_x 本身**。
- 内存释放必须配对：`malloc/free`、`new[]/delete[]`、`cudaMalloc/cudaFree`。


## 3. `cudaMemcpy` 数据传输

```cpp
cudaMemcpy(dst, src, M, kind);  // 目标、来源、字节数、拷贝方向
cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice); // CPU → GPU
cudaMemcpy(h_z, d_z, M, cudaMemcpyDeviceToHost); // GPU → CPU
```

- `cudaMemcpyHostToHost`：CPU → CPU；`cudaMemcpyDeviceToDevice`：GPU → GPU。
- `cudaMemcpyDefault` 可以在支持统一虚拟寻址（UVA）的环境下自动判断拷贝方向。
- 普通的 Host 指针不能直接当作设备数组指针传给 Kernel；通常先把数据复制到 `cudaMalloc` 分配的设备内存中。


## 4. Kernel 与线程索引

```cpp
__global__ void add(const double *x, const double *y, double *z, int N)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return; // 防止数组越界
    z[n] = x[n] + y[n];
}

const int block_size = 128;
const int grid_size = (N + block_size - 1) / block_size; // 向上取整
add<<<grid_size, block_size>>>(d_x, d_y, d_z, N);
```

- 一维全局索引：`n = blockIdx.x * blockDim.x + threadIdx.x`；一个线程对应一个元素。
- `<<<grid_size, block_size>>>` 分别指定 Block 数和每个 Block 的线程数。
- 若 `N` 不是 `block_size` 的整数倍，向上取整会产生多余线程，必须用 `n < N` 或 `if (n >= N) return` 防越界；**不能写成 `n > N`**。
- Kernel 必须用 `__global__` 修饰、返回 `void`，可用 `return;` 提前退出，但不能返回数值。


## 5. `__global__`、`__device__`、`__host__`

- `__global__`：Kernel，通常由 CPU 发起调用，在 GPU 执行，调用时带 `<<<...>>>`。
- `__device__`：设备函数，由 GPU 代码调用、在 GPU 执行；普通调用不写 `<<<...>>>`。
- `__host__`：主机函数，由 CPU 调用并执行，可省略标识符。
- `__host__ __device__`：同一函数同时编译出 CPU 和 GPU 可调用的版本。
- 设备函数可通过**返回值、指针、引用**传出结果：

```cpp
__device__ double f1(double x, double y) { return x + y; }
__device__ void f2(double x, double y, double *z) { *z = x + y; }
__device__ void f3(double x, double y, double &z) { z = x + y; }

// 调用：z[n] = f1(x[n], y[n]);
//       f2(x[n], y[n], &z[n]);
//       f3(x[n], y[n], z[n]);
```

其中 `*z` 是解引用，`&z[n]` 是取地址，函数参数中的 `double &z` 则是引用。


## 6. 结果检查与 API 返回值

- 双精度浮点数可能有舍入误差，使用 `fabs(z[i] - expected) > EPSILON` 判断结果是否超出容差，而不是直接用 `==`。
- `cudaMalloc`、`cudaMemcpy`、`cudaFree` 都返回 `cudaError_t`；成功时为 `cudaSuccess`。如何系统检查错误属于**第 4 章**。


## 最后速记

```text
CPU: malloc / new[]          → free / delete[]
GPU: cudaMalloc              → cudaFree

cudaMalloc((void**)&d_x, M)   → 分配 GPU 显存，修改 d_x 的值
cudaMemcpy(dst, src, M, kind) → 按字节复制数据
HostToDevice                 → CPU → GPU
DeviceToHost                 → GPU → CPU

Kernel: __global__ void f(...)  → <<<grid, block>>> 启动
Device: __device__             → 在 GPU 内调用
Host:   __host__               → 在 CPU 运行

n = blockIdx.x * blockDim.x + threadIdx.x
grid_size = (N + block_size - 1) / block_size
if (n >= N) return;            → 防越界
```
