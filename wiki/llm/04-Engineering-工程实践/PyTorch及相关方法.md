---
date: 2026-04-15
---

# PyTorch 及其他相关方法

关联阅读：可结合 [[10-LLM-大语言模型/02-Pretraining-预训练/LLM预训练底层机制|LLM预训练底层机制]] 理解；PyTorch 方法在预训练循环中形成完整训练系统。


## 1. PyTorch 的基础架构

| 组件 | 一句话理解 |
| --- | --- |
| `Tensor` | 张量，PyTorch 最基础的数据结构。 |
| `Parameter` | 可训练参数，会被模型自动注册。 |
| `Module` | 层或模型的基类，用来组织参数和子模块。 |
| `Autograd` | 自动求导系统，负责计算梯度。 |
| `Optimizer` | 根据梯度更新参数。 |

## 2. 常用 `torch` 方法

### 2.1 `torch.arange`

- **作用**：创建一个一维等差数列张量。
- **常见写法**：`torch.arange(start, end, step)`。
- **适用场景**：生成位置索引、步长序列、范围张量。

```python
import torch

print(torch.arange(0, 10, 2))
# tensor([0, 2, 4, 6, 8])

print(torch.arange(0.0, 1.0, 0.25))
# tensor([0.0000, 0.2500, 0.5000, 0.7500])

print(torch.arange(5, 0, -1))
# tensor([5, 4, 3, 2, 1])
```

### 2.2 `torch.outer`

- **作用**：计算两个一维向量的外积。
- **输入**：两个一维张量。
- **输出**：一个二维矩阵，形状是 `[N, M]`。

```python
import torch

v1 = torch.arange(1, 4)  # tensor([1, 2, 3])
v2 = torch.arange(1, 3)  # tensor([1, 2])

print(torch.outer(v1, v2))
# tensor([[1, 2],
#         [2, 4],
#         [3, 6]])
```

### 2.3 `torch.where`

- **作用**：按条件逐元素选择值，类似张量版 `if / else`。
- **常见写法**：`torch.where(condition, x, y)`。
- **返回**：条件为 `True` 取 `x`，否则取 `y`。

```python
import torch

x = torch.tensor([1, 2, 3, 4, 5])
y = torch.tensor([10, 20, 30, 40, 50])
condition = x > 3

print(torch.where(condition, x, y))
# tensor([10, 20, 30,  4,  5])
```

### 2.4 `torch.cat`

- **作用**：沿指定维度拼接多个张量。
- **常见写法**：`torch.cat([t1, t2], dim=0)`。
- **注意**：除拼接维度外，其他维度必须一致。

```python
import torch

t1 = torch.tensor([[1, 2], [3, 4]])
t2 = torch.tensor([[5, 6], [7, 8]])

print(torch.cat([t1, t2], dim=0))
# tensor([[1, 2],
#         [3, 4],
#         [5, 6],
#         [7, 8]])

print(torch.cat([t1, t2], dim=1))
# tensor([[1, 2, 5, 6],
#         [3, 4, 7, 8]])
```

### 2.5 `torch.cos`

- **作用**：逐元素计算余弦值。
- **输入**：弧度制张量。
- **输出**：形状不变。

```python
import math
import torch

angles = torch.tensor([0, math.pi / 2, math.pi])
print(torch.cos(angles))
# tensor([ 1.0000e+00, -8.7423e-08, -1.0000e+00])
```

### 2.6 `torch.sin`

- **作用**：逐元素计算正弦值。
- **输入**：弧度制张量。
- **输出**：形状不变。

```python
import math
import torch

angles = torch.tensor([0, math.pi / 2, math.pi])
print(torch.sin(angles))
# tensor([0.0000e+00, 1.0000e+00, 1.2246e-16])
```

### 2.7 `torch.triu`

- **作用**：保留矩阵的上三角部分。
- **常见写法**：`torch.triu(input, diagonal=0)`。
- **`diagonal` 含义**：
  - `0`：从主对角线开始保留。
  - `1`：从主对角线上方一条线开始保留。
  - `-1`：会多保留一部分下三角。

```python
import torch

x = torch.tensor([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
])

print(torch.triu(x))
# tensor([[1, 2, 3],
#         [0, 5, 6],
#         [0, 0, 9]])

print(torch.triu(x, diagonal=1))
# tensor([[0, 2, 3],
#         [0, 0, 6],
#         [0, 0, 0]])
```

### 2.8 `torch.reshape`

- **作用**：改变张量形状，但不改变数据内容。
- **常见写法**：`torch.reshape(x, shape)`。
- **注意**：新旧形状的元素总数必须一致。

```python
import torch

x = torch.arange(1, 7)
print(torch.reshape(x, (2, 3)))
# tensor([[1, 2, 3],
#         [4, 5, 6]])

print(torch.reshape(x, (3, -1)))
# tensor([[1, 2],
#         [3, 4],
#         [5, 6]])
```

### 2.9 `torch.topk`

- **作用**：取出某个维度上最大的或最小的前 `k` 个值。
- **常见写法**：`torch.topk(input, k, dim=None, largest=True)`。
- **返回**：
  1. `values`：取出的值。
  2. `indices`：这些值对应的索引。

```python
import torch

x = torch.tensor([10, 30, 20, 40])
values, indices = torch.topk(x, k=2)

print(values)   # tensor([40, 30])
print(indices)  # tensor([3, 1])
```

### 2.10 `torch.repeat_interleave`

- **作用**：按元素级别重复张量中的值。
- **和 `repeat` 的区别**：它重复的是“元素”，不是整段张量。
- **常见场景**：扩展标签、复制索引、构造分组映射。

```python
import torch

x = torch.tensor([1, 2, 3])
print(torch.repeat_interleave(x, repeats=2))
# tensor([1, 1, 2, 2, 3, 3])

repeats = torch.tensor([1, 2, 3])
print(torch.repeat_interleave(x, repeats))
# tensor([1, 2, 2, 3, 3, 3])
```

### 2.11 `torch.argsort`

- **作用**：返回排序后元素在原张量中的索引。
- **直觉**：不直接给你排序结果，而是告诉你“谁排第几”。
- **常见场景**：排序索引、排名、特征重排。

```python
import torch

x = torch.tensor([30, 10, 20])
idx = torch.argsort(x)

print(idx)
# tensor([1, 2, 0])

print(x[idx])
# tensor([10, 20, 30])
```

### 2.12 `torch.bincount`

- **作用**：统计非负整数出现的次数。
- **适用前提**：输入必须是非负整数张量。
- **常见场景**：类别计数、直方图统计。

```python
import torch

x = torch.tensor([0, 1, 1, 3, 2, 1])
print(torch.bincount(x))
# tensor([1, 3, 1, 1])

weights = torch.tensor([1.0, 1.0, 1.0, 2.0, 1.0, 1.0])
print(torch.bincount(x, weights=weights))
# tensor([1., 3., 1., 2.])
```

### 2.13 `torch.div`

- **作用**：逐元素除法。
- **本质**：`torch.div(a, b)` 等价于 `a / b`。
- **适用场景**：张量归一化、缩放、逐元素运算。

```python
import torch

a = torch.tensor([10, 20, 30])
b = torch.tensor([2, 5, 3])

print(torch.div(a, b))
# tensor([ 5.,  4., 10.])
```

### 2.14 `torch.mean`

- **作用**：求平均值。
- **常见写法**：`torch.mean(x, dim=...)`。
- **记忆方式**：`dim` 表示“被消掉的维度”。

```python
import torch

x = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
])

print(torch.mean(x, dim=0))
# tensor([2.5000, 3.5000, 4.5000])

print(torch.mean(x, dim=1))
# tensor([2., 5.])
```

### 2.15 `torch.scatter_add_`

- **作用**：按索引把值加到目标张量指定位置。
- **下划线 `_` 含义**：原地操作，会直接修改原张量。
- **直觉**：把数据按 `index` 分桶相加。

```python
import torch

out = torch.zeros(5)
index = torch.tensor([0, 1, 0, 3])
src = torch.tensor([1.0, 2.0, 3.0, 4.0])

out.scatter_add_(dim=0, index=index, src=src)
print(out)
# tensor([4., 2., 0., 4., 0.])
```

## 3. 常用 `tensor` 方法

### 3.1 `tensor.transpose`

- **作用**：交换两个指定维度。
- **常见写法**：`x.transpose(dim0, dim1)`。
- **常见场景**：注意力里的维度交换，如 `seq_len` 和 `head` 互换。

```python
import torch

x = torch.randn(2, 3, 4)
y = x.transpose(1, 2)
print(y.shape)
# torch.Size([2, 4, 3])
```

### 3.2 `tensor.view`

- **作用**：在不改变数据内容的前提下重塑张量视图。
- **常见写法**：`x.view(new_shape)`。
- **注意**：
  - 新旧形状元素总数必须一致。
  - `view()` 要求张量在内存中是连续的。
  - 如果不确定是否连续，优先用 `reshape()`。

```python
import torch

x = torch.tensor([
    [1, 2, 3, 4, 5, 6],
    [7, 8, 9, 10, 11, 12],
])

print(x.view(3, 4))
# tensor([[ 1,  2,  3,  4],
#         [ 5,  6,  7,  8],
#         [ 9, 10, 11, 12]])

print(x.view(4, 3))
# tensor([[ 1,  2,  3],
#         [ 4,  5,  6],
#         [ 7,  8,  9],
#         [10, 11, 12]])
```

### 3.3 `tensor.unsqueeze`

- **作用**：在指定位置插入一个大小为 `1` 的维度。
- **常见写法**：`x.unsqueeze(dim)`。
- **常见场景**：补 batch 维、补 head 维、方便广播。

```python
import torch

x = torch.tensor([1, 2, 3])
print(x.shape)
# torch.Size([3])

print(x.unsqueeze(0).shape)
# torch.Size([1, 3])

print(x.unsqueeze(1).shape)
# torch.Size([3, 1])
```

## 4. 常用 `nn` 组件

### 4.1 `nn.Parameter`

- **作用**：把一个普通张量包装成“可训练参数”。
- **关键特点**：
  - 会被 `nn.Module` 自动注册到 `.parameters()`。
  - 默认 `requires_grad=True`。
- **适用场景**：自定义权重、偏置、缩放参数。

```python
import torch
import torch.nn as nn

weight = nn.Parameter(torch.ones(4))
print(weight.requires_grad)
# True
```

### 4.2 `nn.Module`

- **作用**：所有层和模型的基类。
- **你通常用它做两件事**：
  - 在 `__init__` 中定义参数和子模块。
  - 在 `forward()` 中定义数据流。
- **额外能力**：支持 `.parameters()`、`.to(device)`、`.train()`、`.eval()`。

```python
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)
```

### 4.3 `nn.Linear`

- **作用**：执行线性变换 `y = xW^T + b`。
- **常见写法**：`nn.Linear(in_features, out_features, bias=True)`。
- **适用场景**：特征映射、维度升降、输出层投影。

```python
import torch.nn as nn

layer = nn.Linear(in_features=10, out_features=5, bias=True)
```

### 4.4 `nn.Dropout`

- **作用**：训练时随机把一部分元素置零，防止过拟合。
- **常见写法**：`nn.Dropout(p=0.5)`。
- **注意**：
  - 只在 `model.train()` 时生效。
  - 在 `model.eval()` 时不会随机丢弃。

```python
import torch.nn as nn

dropout = nn.Dropout(p=0.5)
```

### 4.5 `nn.Embedding`

- **作用**：查找表，把整数 id 映射成向量。
- **常见场景**：词向量、token embedding、类别 embedding。
- **输入要求**：输入通常是整数类型，如 `torch.long`。

```python
import torch
import torch.nn as nn

embedding = nn.Embedding(num_embeddings=1000, embedding_dim=32)
input_ids = torch.tensor([[1, 5, 2, 9]], dtype=torch.long)
output = embedding(input_ids)

print(output.shape)
# torch.Size([1, 4, 32])
```

### 4.6 `nn.ModuleList`

- **作用**：一个能被 PyTorch 正确注册的模块列表。
- **为什么不用普通 list**：普通 Python 列表里的层不会被 `.parameters()` 或 `.to(device)` 自动识别。
- **适用场景**：循环堆叠多层、按需迭代子模块。

```python
import torch.nn as nn

class GoodModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(10, 10),
            nn.Linear(10, 10),
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
```

### 4.7 `nn.Module.register_buffer`

- **作用**：注册一个“随模型一起移动和保存，但不参与训练”的张量。
- **好处**：
  - 会跟随 `model.to(device)` 自动移动。
  - 会进入 `state_dict()`。
  - 不会出现在 `model.parameters()` 中。
- **适用场景**：固定掩码、位置编码表、统计量。

```python
import torch
import torch.nn as nn

class MyAttention(nn.Module):
    def __init__(self, max_len=512):
        super().__init__()
        mask = torch.triu(
            torch.full((max_len, max_len), float('-inf')),
            diagonal=1,
        ).unsqueeze(0).unsqueeze(0)
        self.register_buffer('causal_mask', mask)

    def forward(self, x):
        seq_len = x.shape[2]
        return x + self.causal_mask[:, :, :seq_len, :seq_len]
```

## 5. 训练相关方法与对象

### 5.1 `torch.nn.utils.clip_grad_norm_`

- **作用**：梯度裁剪，防止梯度爆炸。
- **常见写法**：`torch.nn.utils.clip_grad_norm_(parameters, max_norm)`。
- **适用场景**：训练 Transformer、RNN、LLM 时非常常见。

```python
import torch.nn as nn

# 假设 model 已经完成 backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 5.2 `optimizer.param_groups`

- **作用**：保存优化器的参数组和对应超参数。
- **最常见用途**：动态修改学习率。
- **典型操作**：遍历每个参数组，更新其中的 `lr`。

```python
for param_group in optimizer.param_groups:
    param_group['lr'] = lr
```

### 5.3 `scaler.scale`

- **作用**：AMP 混合精度训练中的梯度缩放。
- **为什么需要它**：`float16` 下很小的梯度可能下溢成 0。
- **核心思路**：先把 `loss` 放大，再反向传播，保护微小梯度。

```python
with autocast():
    loss = model(x)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## 6. 快速记忆

| 方法 / 组件 | 最短记忆 |
| --- | --- |
| `torch.arange` | 生成等差数列 |
| `torch.outer` | 向量外积 |
| `torch.where` | 条件选择 |
| `torch.cat` | 沿维度拼接 |
| `torch.cos` / `torch.sin` | 三角函数 |
| `torch.triu` | 取上三角 |
| `torch.reshape` | 改形状 |
| `torch.topk` | 取前 k 个 |
| `torch.repeat_interleave` | 元素级重复 |
| `torch.argsort` | 返回排序索引 |
| `torch.bincount` | 计数统计 |
| `torch.div` | 逐元素除法 |
| `torch.mean` | 求平均 |
| `torch.scatter_add_` | 按索引累加 |
| `tensor.transpose` | 交换两个维度 |
| `tensor.view` | 连续内存上的重塑 |
| `tensor.unsqueeze` | 插入大小为 1 的维度 |
| `nn.Parameter` | 可训练参数 |
| `nn.Module` | 模型 / 层的基类 |
| `nn.Linear` | 线性映射 |
| `nn.Dropout` | 随机丢弃 |
| `nn.Embedding` | 查表取向量 |
| `nn.ModuleList` | 可注册的模块列表 |
| `register_buffer` | 非训练但随模型保存的张量 |
| `clip_grad_norm_` | 梯度裁剪 |
| `optimizer.param_groups` | 调学习率 / 参数组 |
| `scaler.scale` | AMP 梯度缩放 |

## 相关笔记

- **应用**：[[10-LLM-大语言模型/02-Pretraining-预训练/LLM预训练底层机制|LLM预训练底层机制]] — PyTorch 方法在预训练循环中形成完整训练系统。
- **应用**：[[10-LLM-大语言模型/02-Pretraining-预训练/PretrainDataset和pretrain和utils方法|PretrainDataset和pretrain和utils方法]] — PyTorch 基础在数据管线与训练循环中落地。
- **环境**：[[10-LLM-大语言模型/04-Engineering-工程实践/Python环境配置|Python环境配置]] — PyTorch 实验依赖可复现的 Python 与包管理环境。
- **项目应用**：[[50-Projects-项目/02-MiniMind/MiniMind_interview|MiniMind 项目问答]] — MiniMind 的模型、数据和训练循环均由 PyTorch 实现。
- **对比应用**：[[60-Career-求职/03-Algorithms-算法/Python3_ACM输入输出与手撕模板|Python3_ACM输入输出与手撕模板]] — 张量编程与 ACM 手撕题都依赖扎实的 Python 数据结构和输入处理。
