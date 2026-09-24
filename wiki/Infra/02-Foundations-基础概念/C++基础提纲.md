# C++ → CUDA 基础复习提纲

## 1. 指针、引用、数组、指针运算

**它是干什么的：** 直接操作内存地址，是 CUDA 里访问设备内存、传递数组的基础手段。

- 指针保存地址，`*` 解引用取值，`&` 可表示取地址或引用。
- 数组名常退化为首元素指针，`arr[i]` 等价于 `*(arr + i)`。
- 指针 `+1` 按元素大小移动，不是按字节移动。
- 引用是别名，必须初始化、不能改绑，比指针更安全。
- `const int*` 表示内容不可改；`int* const` 表示指针本身不可改。
- CUDA 里 host 指针和 device 指针不能混用，CPU 不能直接解引用普通 `cudaMalloc` 得到的 `d_data`。

---

## 2. `struct`、函数、作用域

**它是干什么的：** 组织数据和逻辑，决定变量在哪可见、活多久，是封装 kernel 参数和资源的基础。

- `struct` 把相关数据打包成新类型，C++ 里和 `class` 的主要区别是默认访问权限不同。
- 函数封装逻辑，参数可用值、指针、引用三种方式传递。
- 作用域决定变量在哪里可见，生命周期决定对象存在多久，两者不是一回事。
- 普通局部变量通常由栈实现，离开对应作用域后生命周期结束；全局/静态对象通常活到程序结束。
- `static` 局部变量只初始化一次，生命周期贯穿程序。
- CUDA kernel 参数常可用 `struct` 组织；`__shared__` 变量由同一个 block 内的线程共享。

---

## 3. `std::vector`、`std::unique_ptr`

**它是干什么的：** 自动管理动态内存，避免手动 `new` / `delete` 带来的泄漏和崩溃。

- `vector` 是自动扩容的动态数组，内部通常维护数据指针、`size` 和 `capacity`。
- `reserve` 可以预留容量，减少反复扩容；扩容后原来的指针、引用和迭代器可能失效。
- `data()` 返回底层连续内存的裸指针；普通 `std::vector` 的 `data()` 是 host 指针，可作为 `cudaMemcpy` 的 host 端地址使用。
- `unique_ptr` 是独占所有权的智能指针，不能复制，只能移动。
- `std::make_unique` 比直接使用 `new` 更安全，现代 C++ 中优先使用。
- `unique_ptr` 可以配合自定义删除器管理特殊资源，例如封装 `cudaMalloc` / `cudaFree`。
- `vector` 和 `unique_ptr` 都体现了 RAII：对象离开作用域时自动释放资源。

---

## 4. 模板基础

**它是干什么的：** 用一套代码适配多种类型或编译期常量，是 STL 和泛型 CUDA kernel 的底层机制。

- `template <typename T>` 让编译器根据类型生成具体版本。
- `std::vector<int>`、`std::unique_ptr<float>` 都是模板实例化后的具体类型。
- 模板实例化时编译器必须看到完整定义，因此模板通常放在头文件或 `.cuh` 中。
- 非类型模板参数如 `template <int BLOCK_SIZE>` 可以在编译期确定数组大小或配置。
- CUDA 中常用模板编写泛型 kernel、固定 block 大小、做类型分派和编译期优化。
- 模板报错通常很长，定位时优先看最前面的直接错误和最底层的实例化位置。

---

## 5. 头文件、源文件、编译链接

**它是干什么的：** 把声明和实现分离，让多个源文件共享接口，最后由链接器合并成可执行程序或库。

- `.h` / `.cuh` 通常放声明，`.cpp` / `.cu` 通常放实现。
- `#include` 可以理解为预处理阶段把头文件内容展开到当前文件中；`#pragma once` 用于防止重复包含。
- 编译通常以源文件为单位生成目标文件，链接阶段再把多个目标文件和库合并。
- 编译错误通常关注语法、类型和声明；链接错误通常关注定义缺失、符号重复和库未链接。
- `undefined reference` 往往意味着实现没有参与链接，或者缺少相应库。
- 含有 CUDA 扩展语法（如 `__global__`、`<<< >>>`）的代码需要经过 CUDA 编译流程。
- 工程里经常由 CMake 统一组织 C++ 和 CUDA 源文件的编译、链接。

---

## 6. RAII

**它是干什么的：** 把资源生命周期绑定到对象生命周期，构造时获取资源，析构时释放资源，让资源管理自动化、异常安全。

- 核心思想：构造时申请，析构时释放，离开作用域自动清理。
- `std::vector`、`std::string`、`std::unique_ptr`、`std::ifstream` 都是 RAII 的典型例子。
- CUDA 中可以把 `cudaMalloc` / `cudaFree`、`cudaStreamCreate` / `cudaStreamDestroy` 等封装到 RAII 类中。
- 对于独占资源类，通常禁止拷贝，避免多个对象释放同一资源。
- 独占资源类通常允许移动，以便安全转移所有权。
- 析构函数通常不应抛异常，常配合 `noexcept`。
- 即使中途 `return` 或发生异常，已经构造完成的局部 RAII 对象仍会执行析构，降低资源泄漏风险。

---

## 7. CMake

**它是干什么的：** 用一份配置描述项目结构，跨平台生成构建系统，管理源文件、依赖和编译选项。

- `CMakeLists.txt` 是项目构建配置文件。
- 常见流程是先运行 `cmake` 生成构建文件，再使用 `cmake --build` 或对应构建工具完成编译。
- `project(... LANGUAGES CXX CUDA)` 可以启用 C++ 和 CUDA 支持。
- `add_executable` / `add_library` 用来定义构建目标。
- `target_include_directories` 配置头文件搜索路径。
- `target_link_libraries` 用于链接依赖库。
- `CUDA_ARCHITECTURES` 可用于指定目标 GPU 架构。
- 通常使用独立 `build/` 目录保存生成文件，保持源码目录整洁。
- 中大型 CUDA 项目通常使用 CMake 统一管理构建。

---

## 8. CUDA 错误检查

**它是干什么的：** 及时发现 CUDA API 和 kernel 的错误，避免静默失败和错误结果。

- CUDA API 调用和 kernel 执行都可能失败，需要主动检查返回值或错误状态。
- 常见做法是封装 `CHECK_CUDA` 检查 CUDA API，封装 `CHECK_KERNEL` 检查 kernel。
- `cudaGetLastError()` 常用于检查 kernel 启动相关错误。
- `cudaDeviceSynchronize()` 可以让 CPU 等待 GPU 完成，从而暴露 kernel 执行期错误。
- 常见错误包括 `cudaErrorMemoryAllocation`、`cudaErrorIllegalAddress`、`cudaErrorLaunchOutOfResources`。
- 调试时可使用 `compute-sanitizer`、`CUDA_LAUNCH_BLOCKING=1` 等工具或环境变量辅助定位问题。
- 生产环境中应谨慎使用频繁的 `cudaDeviceSynchronize()`，否则会破坏 CUDA 的异步执行和并行性。

---

