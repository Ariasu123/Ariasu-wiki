
### 1. 为什么需要归一化？

关联阅读：可结合 [MiniMind架构图解析](MiniMind%E6%9E%B6%E6%9E%84%E5%9B%BE%E8%A7%A3%E6%9E%90.md) 理解；整体架构给出了归一化层的调用位置。


大模型包含成百上千个 Linear 层。在反向传播计算梯度时，公式所示：

$$\frac{dL}{dW} = \frac{dL}{dY} \cdot X$$

这意味着：**梯度的大小直接受到输入值 $X$ 的影响。**

- 如果 $X$ 的数值非常大，梯度就会“爆炸”（变得无穷大）。
    
- 如果 $X$ 的数值非常小，梯度就会“消失”（模型学不动）。

为了让训练稳如泰山，我们需要把 $X$ “约束”在一个合理的范围内，这就是**归一化**。

---
### 2. 从 LayerNorm 到 RMSNorm

传统的 **LayerNorm (LN)** 认为，要把一组数据变规范，需要做两件事：

1. **平移**：减去均值 $\mu$，让数据的中心回到 0。
    
2. **缩放**：除以标准差 $\sigma$，让数据的波动范围变统一。

**但是，RMSNorm 的作者发现：** 真正起到稳定梯度作用的主要是“缩放”**，而“平移”计算均值的过程对模型效果贡献不大，反而增加了计算开销。

---

### 3. RMSNorm 的数学公式拆解

RMSNorm 的核心思想是：**只利用“均方根”进行缩放。**

#### 第一步：计算均方根 (RMS)

对于一个输入的向量 $x = (x_{1}, x_{2}, ..., x_{n})$，先计算它所有元素的平方平均值的平方根：

$$RMS(x) = \sqrt{\frac{1}{n} \sum_{i=1}^{n} x_{i}^{2} + \epsilon}$$

- **$n$**：向量的维度。
- **$\epsilon$ (Epsilon)**：一个极小的数（如 $1e-5$），防止分母为 0 导致报错。

#### 第二步：标准化

把原始的 $x_{i}$ 除以这个 RMS，得到标准化的值：

$$\bar{x}_{i} = \frac{x_{i}}{RMS(x)}$$

#### 第三步：可学习的缩放 (Scaling)

为了不让模型的能力被死死限制在特定范围内，我们会给它一个“弹簧”——可学习的参数 $\gamma$。更严谨地说，它通常是按维度学习的一组参数 $\gamma_{i}$：

$$y_{i} = \bar{x}_{i} \cdot \gamma_{i}$$

- **$\gamma_{i}$ (Gamma)**：模型在训练过程中会自动调整这个值，找到最适合当前维度的缩放比例。

对应的 Python 实现如下：

```python
class RMSNorm(torch.nn.Module):
    """
    RMS归一化 (Root Mean Square Normalization)
    相比LayerNorm，RMSNorm去掉了均值中心化，只保留方差缩放
    计算更简单，效果相当，在大模型中广泛使用
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        """
        Args:
            dim: 归一化的维度大小
            eps: 防止除零的小常数
        """
        super().__init__()                              # 调用父类nn.Module构造函数
        self.eps = eps                                  # 存储epsilon值
        # nn.Parameter: 将tensor注册为可学习参数，会自动加入optimizer
        # torch.ones(dim): 创建全1的tensor作为缩放参数
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        RMSNorm的核心计算：x / sqrt(mean(x^2) + eps)
        """
        # x.pow(2): 对x每个元素平方
        # .mean(-1, keepdim=True): 在最后一维求均值，保持维度
        # torch.rsqrt(): 计算平方根的倒数，即 1/sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入tensor，shape为[batch, seq_len, dim]
        Returns:
            归一化后的tensor
        """
        # .float(): 转换为float32进行计算，提高数值稳定性
        # .type_as(x): 将结果转换回x的原始数据类型
        # self.weight *: 可学习的缩放参数
        return self.weight * self._norm(x.float()).type_as(x)
```

### 总结

**RMSNorm 就是 LayerNorm 的“极致精简版”。** 它抛弃了“减去均值”的操作，认为只要通过除以**均方根**来约束输入的量级，就足以解决梯度爆炸和消失的问题。对于像 MiniMind 这种追求效率的模型，这种“又快又稳”的数学结构是最佳选择

## 相关笔记

- **总览**：[MiniMind架构图解析](MiniMind%E6%9E%B6%E6%9E%84%E5%9B%BE%E8%A7%A3%E6%9E%90.md) — 整体架构给出了归一化层的调用位置。
- **训练作用**：[LLM预训练底层机制](../02-Pretraining-%E9%A2%84%E8%AE%AD%E7%BB%83/LLM%E9%A2%84%E8%AE%AD%E7%BB%83%E5%BA%95%E5%B1%82%E6%9C%BA%E5%88%B6.md) — 归一化层直接影响深层网络的数值稳定性。
