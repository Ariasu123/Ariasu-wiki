![](../_assets/Architecture-%E6%A8%A1%E5%9E%8B%E6%9E%B6%E6%9E%84/Rope%E5%92%8CYaRN.png)

**RoPE (Rotary Positional Embedding，旋转位置编码)** 是目前大模型（如 Llama, Qwen, MiniMind）的标配。

在它出现之前，模型很难处理超长的句子。RoPE 的核心天才之处在于：**它利用复数旋转的数学特性，把“绝对位置”变成了“相对位置”。**

---

### 1. 为什么要有 RoPE？（痛点）

关联阅读：可结合 [MiniMind架构图解析](MiniMind%E6%9E%B6%E6%9E%84%E5%9B%BE%E8%A7%A3%E6%9E%90.md) 理解；架构笔记提供位置编码所在的模型上下文。


在 Transformer 中，如果不加位置编码，模型看“我爱吃蝉”和“吃我爱蝉”是一样的。

- **早期的做法（绝对位置）**：给每个位置一个固定的标签（比如第 1 个词加 0.1，第 2 个加 0.2）。
- **缺点**：如果训练时只见过 512 长度的句子，碰到 1024 长度时，模型就不知道 512 之后的标签是什么意思了，外推性很差。
---

### 2. RoPE 的数学直觉：旋转

RoPE 的做法是：**根据词所在的位置 $m$，把这个指针旋转一个特定的角度 $m\theta$。**

- 第 1 个词：旋转 $\theta$ 度。
- 第 2 个词：旋转 $2\theta$ 度。
- 第 $m$ 个词：旋转 $m\theta$ 度。
---

### 3. 核心数学推导

RoPE 想要实现一个目标：当模型计算两个词 $q_m$（位置 $m$）和 $k_n$（位置 $n$）的注意力分数时，结果只和它们的**相对距离 $m-n$** 有关。

#### 复数表示

在二维平面上，我们可以把词向量 $(x, y)$ 看作一个复数 $x + iy$。旋转操作可以写成：

$$f(x, m) = x \cdot e^{im\theta}$$

这里 $e^{im\theta}$ 就是旋转算子。

#### 为什么有效？

当你计算 $q$ 和 $k$ 的点积（即注意力）时：

$$Score \propto \text{Re}[q_{m} \cdot k_{n}^{*}] = \text{Re}[(q e^{im\theta}) \cdot (k e^{in\theta})^{*}] = \text{Re}[q \cdot k^{*} \cdot e^{i(m-n)\theta}]$$

注意力最终与 $(m-n)\theta$有关。

**结论**：模型不再死记硬背“我在第 5 个位置”，而是通过向量之间的**夹角**，直接感知到“你离我 3 个单位远”。

---

### 4. 具体的计算实现

在代码实现中，我们不会真的用复数，而是使用一个**旋转矩阵**。

对于一个高维向量，RoPE 会把它两两成对，每两个维度组成一个二维平面进行旋转：

$$\begin{pmatrix} q_0 \\ q_1 \end{pmatrix} \rightarrow \begin{pmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{pmatrix} \begin{pmatrix} q_0 \\ q_1 \end{pmatrix}$$
#### 1. 数学本质：复数旋转的等价变换

RoPE 利用复数乘法 $(a + bi) \cdot e^{i\theta}$ 来实现向量旋转。其展开结果为：
- **新实部**：$a\cos\theta - b\sin\theta$
- **新虚部**：$b\cos\theta + a\sin\theta$
#### 2. 代码：`rotate_half` (实数模拟复数)

为了避免昂贵的复数运算，代码将 512 维向量平分为**左半段 (实部 $a$)** 和**右半段 (虚部 $b$)**。
- **原始向量 $q$**：$[a, b]$
- **`rotate_half(q)`**：$[-b, a]$ （等效于复数乘以虚数单位 $i$）
#### 3. 最终组合公式

代码通过一次性矩阵运算还原复数旋转：
$$q_{embed} = (q \cdot \cos) + (\text{rotate-half}(q) \cdot \sin)$$
---

为了覆盖不同频率的信息，每个维度的旋转速度 $\theta$ 是不一样的：
$$\theta_i = \text{base}^{-2i/d}$$

#### 1. 数学定义：从频率到波长

在 RoPE 公式中，第 $i$ 组维度的旋转频率是 $\theta_i$。

数学上，旋转一圈（$2\pi$ 弧度）所需的距离 $L$ 就是波长：

$$L_i = \frac{2\pi}{\theta_i}$$

由于 $\theta_i = \text{base}^{-2i/d}$，代入后得到波长公式：

$$L_i = 2\pi \cdot \text{base}^{\frac{2i}{d}}$$

---

#### 2. 直观理解：快表与慢表

模型里有成百上千个维度，它们的波长各不相同：

- **高频维度（$i$ 较小）**：
    
    - **波长短**：比如波长为 10。
    - **含义**：每走 10 个字，指针就转满一圈。
    - **作用**：它对位置极其敏感。只要词稍微挪动一个位置，角度就会变很大。这让模型能看清**微观语法**（比如“的”字后面必须跟着名词）。
        
- **低频维度（$i$ 较大）**：

    - **波长长**：比如波长为 10000。
    - **含义**：走完 1 万个字，指针才勉强转完一圈。
    - **作用**：它对位置极其不敏感。即使两个词隔了几百个字，它们的角度差也很小。这让模型能感知**宏观语义**（比如文章开头提到了一个“伏笔”，结尾能接上）。
---

#### 3. 波长如何决定 YaRN 的逻辑？

YaRN 论文之所以引入波长，是因为它发现：**如果波长相对于“训练长度”太短，模型就不需要插值。**

假设模型是在 $L_{\text{train}} = 2048$ 的长度下训练的：

- **情况 A：波长 $L_i < L_{\text{train}}$（高频区）**
    
    - 比如波长只有 50。在 2048 的训练长度内，这个指针已经足足转了 40 多圈了。
    - 模型已经见过这个指针转到任何角度的样子。所以当你扩展到 32768 时，这个维度的旋转角度依然在模型见过的“老圈子”里打转。
    - **YaRN 处理：** 别动它（$ramp = 0$），保持原速。
        
- **情况 B：波长 $L_i > L_{\text{train}}$（低频区）**
    
    - 比如波长是 10000。在 2048 的长度内，指针只转了 1/5 圈。
    - 模型只见过这 1/5 圈的角度。如果你不改频率，直接读到 8000 位，指针会转到 4/5 圈的位置。
    - 模型惊呼：“这是什么新角度？我受训时没见过！”
    - **YaRN 处理：** 减速（$ramp = 1$）。把它的旋转速度调慢，让它在 8192 位时，正好才转完那见过的 1/5 圈。
##### 1. $d$ 代表什么？（总体长度）
- **含义**：**Model Dimension**（模型的总维度）
##### 2. $i$ 代表什么？（当前位置）
- **含义**：**Dimension Index**（维度索引/下标）。
- **解释**：由于 RoPE 是把高维向量“两两成对”进行旋转的（就像图中 $q_0, q_1$ 组成一对），$i$ 的取值范围是从 $0$ 到 $d/2 - 1$。
- 当 $i=0$ 时，计算的是向量最开始的那两个维度。
- 当 $i$ 变大时，计算的是向量靠后的那些维度。
---

## YaRN

简单来说，RoPE 虽然优雅，但它有一个致命弱点：**训练时模型只见过旋转 0 到 512 度的向量，如果你突然给它一个旋转了 2048 度的向量（长文本），模型就会彻底“晕车”，输出乱码。**

**YaRN** 的出现就是为了解决这个问题，让模型在不增加（或极少增加）微调成本的情况下，处理比训练时**长 10 倍甚至百倍**的文本。

---
### 1. 核心矛盾：分子的“挤压”与“丢失”

当我们要把长度从 2048 扩展到 8192 时，最简单的办法是把旋转角度“压缩”：原来第 8192 个位置的角度，现在强行缩放回第 2048 个位置的角度。

- **问题：** 这样做会导致原本清晰的“相对距离”变得模糊。就像把原本 1 米一个的刻度，强行挤成 25 厘米一个，模型分不清两个邻近的词了。
---
### 2. YaRN 的三个天才改进

YaRN 的数学理论核心是**分频治之**。它意识到：RoPE 中不同维度的旋转频率（$\theta$）代表了不同的语义信息。

#### (1) 高频不插值（保持精细度）

- **理论：** 旋转非常快的维度（高频），负责捕捉**相邻词**的关系。
- **做法：** 对于这些维度，YaRN **完全不动它**。因为相邻词的关系在长短文本中是一样的，动了反而会毁掉模型最基础的语言能力。
#### (2) 低频全插值（扩展范围）

- **理论：** 旋转非常慢的维度（低频），负责捕捉**长距离**的宏观关系。
- **做法：** 对于这些维度，进行平滑的线性缩放。因为长距离信息比较“粗放”，缩放一点点不影响大局，却能让模型感知到更远的位置。
#### (3) 中频混合（波浪式过渡）

- **做法：** 在高频和低频之间，YaRN 使用了一个平滑的函数进行过渡，确保模型在处理不同距离的信息时不会出现“断层”。
----

划分高低维度的索引 $i$，我们需要从 **“波长比例 $b$”** 的定义出发，一步步反推出它在向量维度中的位置。这个推导过程的核心逻辑是：**找到一个临界点，在这个点上，维度的波长正好是训练长度 $L$ 的若干分之一。**

### 1. 基础公式准备

首先，我们已知 RoPE 的波长公式（即旋转一圈所需的距离）：

$$L_i = 2\pi \cdot \text{base}^{\frac{2i}{d}}$$

论文中定义的波长比例 $b$（即在训练长度 $L$ 内转了多少圈）：

$$b = \frac{L}{L_i}$$

---

### 2. 推导划分高低纬度的索引 i 步骤

我们的目标是：已知一个比例 $b$（比如 $\beta_{fast}$ 或 $\beta_{slow}$），求出它对应的维度索引 $i$。

#### 第一步：代入波长公式

将 $L_i$ 的表达式代入 $b$ 的定义中：

$$b = \frac{L}{2\pi \cdot \text{base}^{\frac{2i}{d}}}$$

#### 第二步：移项，隔离底数部分

我们要把带 $i$ 的部分挪到一边：

$$\text{base}^{\frac{2i}{d}} = \frac{L}{b \cdot 2\pi}$$

#### 第三步：左右两边取对数（$\ln$）

为了把指数上的 $i$ 降下来，我们取自然对数：

$$\ln\left(\text{base}^{\frac{2i}{d}}\right) = \ln\left(\frac{L}{b \cdot 2\pi}\right)$$

利用对数性质 $\ln(x^n) = n \ln(x)$：

$$\frac{2i}{d} \cdot \ln(\text{base}) = \ln\left(\frac{L}{b \cdot 2\pi}\right)$$

#### 第四步：解出 $i$

最后，把左边的系数移到右边：

$$i = \frac{d \cdot \ln\left(\frac{L}{b \cdot 2\pi}\right)}{2 \ln(\text{base})}$$
---
我们要计算一个新的频率 $f'(i)$。

$$f'(i) = (1 - \gamma)f(i) + \gamma \frac{f(i)}{s}$$

提取公因子 $f(i)$ 后得到：

$$f'(i) = f(i) \cdot \left( 1 - \gamma + \frac{\gamma}{s} \right)$$

- **当 $\gamma=0$（高频区）**：系数是 1，频率不变。
- **当 $\gamma=1$（低频区）**：频率变为 $1/s$。这意味着旋转速度变慢了 $s$ 倍，原本要在 8192 位转完的角度，现在要在 32768 位才转完。
### 3. 关键绝招：注意力重缩放 (Attention Scaling)

$$\text{cos}' = \cos \cdot \text{attn-factor}$$

**推导逻辑：**

1. **点积膨胀**：在长文本中，由于参与计算的词变多了，$Q \cdot K$ 的平均值会发生偏移。
2. **熵的流失**：数值变大导致 Softmax 后的分布变得太“尖”，模型只关注极少数词。
3. **补偿**：YaRN 通过 `attn_factor`（通常是基于 $\sqrt{t}$ 算出的一个大于 1 的系数）直接给 $\cos$ 和 $\sin$ “加成”。
4. **最终效果**：在计算 $Q \cdot K$ 时，这个系数会平方（因为 $Q$ 和 $K$ 都带了），从而拉大或缩小得分的差距，起到类似“降温”的作用，让注意力重新分布均匀。

当你强行拉长文本长度时，Softmax 的分布会变得非常集中（熵值降低），导致模型变得“呆滞”。YaRN 引入了一个 **温度因子 $\sqrt{t}$**：

$$Score = \text{Softmax}\left(\frac{Q \cdot K}{\sqrt{d} \cdot \mathbf{scale}}\right)$$

- **作用：** 通过引入随长度扩展而增大的缩放因子（scale），抵消长序列导致的注意力点积膨胀，恢复 Softmax 分布的平滑度，确保模型对全局信息的敏锐度。
----


```python
def precompute_freqs(
    dim: int,
    end: int = int(32 * 1024),
    rope_base: float = 1e6,
    rope_scaling: Optional[dict] = None,
):
    # 1. 初始化标准 RoPE 频率。
    # torch.arange(0, dim, 2) 生成 [0, 2, 4, ... dim-2]
    # 计算出的 freqs 就是标准的 1 / (base ** (2i / d))
    freqs, attn_factor = (
        1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)),
        1.0,
    )

    if rope_scaling is not None:
        # 2. 从配置字典中提取 YaRN 的超参数
        # orig_max: 模型预训练时的原始最大长度（例如 Llama-2 是 2048 或 4096）
        # factor: 要扩展的倍数 s (比如从 2k 扩展到 32k，factor 就是 16)
        # beta_fast (对应论文中的 α): 高频边界，波长比例大于此值的维度不缩放
        # beta_slow (对应论文中的 β): 低频边界，波长比例小于此值的维度全量缩放
        # attn_factor: 注意力温度补偿，由于距离拉长导致注意力分布发散（变平缓），需要乘上一个系数让注意力重新“聚焦”
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048),
            rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0),
            rope_scaling.get("beta_slow", 1.0),
            rope_scaling.get("attention_factor", 1.0),
        )

        # 只有当要推断的长度大于原始训练长度时，才应用缩放
        if end / orig_max > 1.0:
            # 3. 使用前文推导的公式，定义波长比例 b 到维度索引 i 的映射函数
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (
                2 * math.log(rope_base)
            )

            # 4. 计算高频区和低频区的维度切分点
            # low: 不需要缩放的高频部分的最高索引
            # high: 需要完全缩放的低频部分的最低索引
            low, high = (
                max(math.floor(inv_dim(beta_fast)), 0),
                min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1),
            )

            # 5. 计算混合因子 γ (Ramp)
            # 在 low 之前，ramp 为 0；在 high 之后，ramp 为 1；在 low 和 high 之间，线性过渡。
            # clamp 函数限制了数值只能在 [0, 1] 之间。
            ramp = torch.clamp(
                (torch.arange(dim // 2, device=freqs.device).float() - low)
                / max(high - low, 0.001),
                0,
                1,
            )

            # 6. 频率融合公式：f'(i) = f(i) * ((1-γ) + γ/s)
            # 当 ramp=0 时（高频）：系数为 1，保持原频率不变。
            # 当 ramp=1 时（低频）：系数为 1/factor，即对频率进行线性插值缩放。
            # ramp在0-1之间时：平滑过渡。
            freqs = freqs * (1 - ramp + ramp / factor)

    # 7. 根据目标长度 end，生成位置索引向量 t
    t = torch.arange(end, device=freqs.device)

    # 8. 计算外积：将位置 t 与处理好的频率 freqs 相乘，得到每个位置的旋转角度 θ
    freqs = torch.outer(t, freqs).float()

    # 9. 计算 Cos 和 Sin，并应用注意力补偿系数 (attn_factor)
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor

    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    def rotate_half(x):
        return torch.cat(
            (-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1
        )

    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (
        rotate_half(q) * sin.unsqueeze(unsqueeze_dim)
    )
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (
        rotate_half(k) * sin.unsqueeze(unsqueeze_dim)
    )
    return q_embed, k_embed
```

## 相关笔记

- **总览**：[MiniMind架构图解析](MiniMind%E6%9E%B6%E6%9E%84%E5%9B%BE%E8%A7%A3%E6%9E%90.md) — 架构笔记提供位置编码所在的模型上下文。
- **应用**：[Attention 与 GQA](Attention%20%E4%B8%8E%20GQA.md) — RoPE 的实际作用体现在 Attention 分数计算中。
- **系统影响**：[vLLM／SGLang 路线](../../ai-infra/01-Roadmaps-%E5%AD%A6%E4%B9%A0%E8%B7%AF%E7%BA%BF/%E8%B7%AF%E7%BA%BF.md) — 长上下文外推会提高 KV Cache 和调度压力。
