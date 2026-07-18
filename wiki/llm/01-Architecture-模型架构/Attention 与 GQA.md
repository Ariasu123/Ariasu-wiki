---
date: 2026-04-17
---

## 1. 注意力机制：

关联阅读：可结合 [[10-LLM-大语言模型/01-Architecture-模型架构/MiniMind架构图解析|MiniMind架构图解析]] 理解；架构图说明了 GQA 在完整模型中的位置。


### 数学公式

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

- **$Q$ (Query)**: “我要找什么？”（当前词的查询向量）。
- **$K$ (Key)**: “我有什么标签？”（所有词的索引向量）。
- **$V$ (Value)**: “我包含什么信息？”（所有词的内容向量）。
- **$\sqrt{d_k}$**: 缩放因子，防止点积结果过大导致梯度消失。
- **$\text{softmax}$**: 将得分转换为概率（权重），总和为 1。

---

## 2. KV Cache：推理的加速器

**为什么需要它？**

在生成模式（Decoding）下，模型是一个字一个字往外蹦的。如果不存缓存，生成第 100 个词时，需要重新计算前 99 个词的 $K$ 和 $V$，这会造成巨大的重复计算。

- **原理**：在计算当前词时，把算好的 $K$ 和 $V$ 存入显存。下次计算只需算“新词”的 `QKV`，然后直接读取历史的 $K$ 和 $V$ 进行拼接。
- **代价**：显存占用极高。显存容量限制了 Batch Size 和最大序列长度。

---

## 3. 三大架构：MHA vs MQA vs GQA

这三种架构的区别本质上是：**Q 和 KV 的配对比例。**

### A. MHA (Multi-Head Attention) - 多头注意力

- **结构**：$H$ 个 $Q$ 对应 $H$ 个 $K$ 和 $H$ 个 $V$（1:1 配对）。
- **优点**：表达能力最强，精度最高。
- **缺点**：**KV Cache 巨大**。每个头都要存一套 KV，显存压力极大。

### B. MQA (Multi-Query Attention) - 多查询注意力

- **结构**：$H$ 个 $Q$ 共享 **1 个** $K$ 和 **1 个** $V$（N:1 配对）。
- **优点**：KV Cache 极小，推理速度极快。
- **缺点**：精度损失大，模型容易变得“智力下降”。

### C. GQA (Grouped-Query Attention) - 分组查询注意力

- **结构**：将 $H$ 个 $Q$ 分成 $G$ 组，每组共享一对 $K$ 和 $V$（N:G 配对）。
- **特点**：例如 32 个 Q，8 个 KV，每 4 个 Q 共用 1 对 KV。
- **意义**：**完美的折中**。显存占用远低于 MHA，但模型表现几乎持平 MHA。

---

## 4. 维度对比表 (以 Llama-7B 为例)

假设 $Batch=1, Seq\_len=1, Head\_dim=128$：

|**架构**|**Q 的形状**|**K/V 的形状 (KV Cache)**|**备注**|
|---|---|---|---|
|**MHA**|`[1, 32, 1, 128]`|`[1, 32, Total, 128]`|每个头都有一份缓存|
|**MQA**|`[1, 32, 1, 128]`|`[1, 1, Total, 128]`|所有头共用一份缓存|
|**GQA**|`[1, 32, 1, 128]`|`[1, 8, Total, 128]`|4 个 Q 头共用一份缓存|

## 5. 在 MiniMind 中 Attention 机制的具体实现

### 1. 输入 Tensor `x`

- 形状：`[bsz, seq_len, hidden_size]`

### 2. Q、K、V 进行线性变换，扩展成多头（四维）

- 使用 `nn.Linear` 分别对 `Q`、`K`、`V` 做线性投影。
- 输入维度是 `hidden_size`，输出维度通常是 `num_head * head_dim`。
- 再通过 `tensor.view（）` 扩展成多头所需的四维张量。

### 3. 对 Q、K 进行 RoPE 位置编码

- `cos` 和 `sin` 的形状为 `[seq_len, head_dim]`。
- 在 `apply_rotary_pos_emb` 的实现中，`cos` 和 `sin` 会在维度 1 上扩展，以匹配 `Q` 和 `K` 的形状：`[seq_len, 1, head_dim]`。
- PyTorch 的广播计算遵循从右往左对齐的规则。

### 4. KV Cache 处理

- 如果提供了 `past_key_value`，就将新的 `K`、`V` 与缓存中的 `K`、`V` 连接起来。
- 拼接发生在 `seq_len` 这个维度上，通常使用 `torch.cat`。

### 5. GQA：进行 KV 重复，以支持多头共享

```python
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    # 提取张量的四个维度
    bs, slen, num_key_value_heads, head_dim = x.shape

    # 防呆设计
    if n_rep == 1:
        return x

    # 在 num_key_value_heads 维度上增加一个维度，再展开并 reshape
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, num_key_value_heads, n_rep, head_dim)
        .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )
```

- 之后通常会通过 `transpose(1, 2)` 交换 `seq_len` 和 `num_head`。
- 这是因为 PyTorch 的注意力矩阵运算主要发生在最后两维。

### 6. Attention 计算

- 满足条件时，优先使用 `flash_attention`，即调用 `torch` 的 `scaled_dot_product_attention`。
- 否则走标准实现：`scores = QK^T / sqrt(d_k)`。
- `scores` 的形状为 `[bsz, heads, seq_len, total_len]`。

#### 注意力掩码

1. 内部掩码（因果掩码）
   - 作用：让模型无法看到后面的词向量信息。
   - 选中 `scores` 张量中最后 `seq_len` 个位置。
   - 使用 `torch.triu` 创建上三角矩阵。
   - 再通过填充把对应位置设为 `-inf`。
   - 这样右上角元素会被设置为 `-inf`，左下角元素保留为 `0`。

2. 外部掩码
   - 作用：屏蔽 `[PAD]` 或逻辑上不该关注的块。
   - `attention_mask` 的形状为 `[bsz, total_len]`。

### 7. 归一化以及 Output 形状恢复（三维）

- 对 `scores` 的 `total_len` 维度进行归一化，再做 `dropout`。
- `output = scores @ V`，此时 `output` 的形状为 `[bsz, n_heads, seq_len, head_dim]`。
- 之后先 `transpose(1, 2)`，再 `reshape` 为 `[bsz, seq_len, n_heads * head_dim]`。
- 最后通过线性层投影（`Linear`）回 `hidden_size` 维度。

### 8. 返回 `output` 和 `past_kv`

- `output` 用于后续的残差连接。
- `past_kv` 用于累积 KV 缓存，支持后续增量解码。

## 代码实现
```python
# 前向传播方法
def forward(
    self,
    x: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    # 过去词的 kv 缓存
    past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    use_cache=False,
    attention_mask: Optional[torch.Tensor] = None,
):
    # 得到输入的 batch size 和序列长度
    bsz, seq_len, _ = x.shape

    # 线性变换得到 q、k、v
    xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)

    # 将 q、k、v 重塑为多头格式
    xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
    xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
    xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

    # 取出 cos、sin 表
    cos, sin = position_embeddings

    # 应用 Rope 位置编码，cos 和 sin 的形状为 [seq_len, head_dim]
    # apply_rotary_pos_emb 的实现中，cos 和 sin 在维度 1 位置被扩展以匹配 q 和 k 的形状
    xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

    # ------------ kv cache 处理 -------------------
    # 如果提供了 past_key_value，则将新的 k 和 v 与缓存的 k 和 v 连接起来
    if past_key_value is not None:
        xk = torch.cat([past_key_value[0], xk], dim=1)
        xv = torch.cat([past_key_value[1], xv], dim=1)

    # 如果需要缓存，返回拼接的 k 和 v
    past_kv = (xk, xv) if use_cache else None

    # ----------- GQA：进行 kv 重复以支持多头共享 ----------
    # 进行矩阵旋转，torch 只支持最后两个维度进行矩阵乘法
    # 所以需要先把 n_local_heads 和 head_dim 交换位置
    xq = xq.transpose(1, 2)  # [bsz, n_local_heads, seq_len, head_dim]
    xk = repeat_kv(xk, self.n_rep).transpose(1, 2)
    xv = repeat_kv(xv, self.n_rep).transpose(1, 2)

    # ------------ Attention 计算 -------------------
    # 优先使用 flash attention，如果满足条件则调用 torch 的 scaled_dot_product_attention
    # 否则使用传统的点积注意力计算
    if (
        self.flash  # 条件一：初始化时启用了 flash attention
        and (seq_len > 1)  # 条件二：序列长度大于 1（长度为 1 时不需要计算注意力）
        and (past_key_value is None)  # 条件三：没有提供 past_key_value（全量计算模式 prefill）
        # 条件四：没有提供 attention_mask，或者提供的 attention_mask 全为 1
        and (attention_mask is None or torch.all(attention_mask == 1))
    ):
        output = F.scaled_dot_product_attention(
            xq,
            xk,
            xv,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,  # 自动开启因果掩码
        )
    else:
        # 标准实现 scores = QK^T / sqrt(d_k)
        # scores 的形状为 [bsz, heads, seq_len, total_len]
        scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # ------ 应用注意力掩码 ------
        # 内部掩码（因果掩码）
        # scores[:, :, :, -seq_len:] 选中了 scores 张量中最后 seq_len 个位置的所有分数
        # torch.triu 创建一个上三角矩阵，用于生成因果掩码
        # 右上角的元素被设置为 -inf，左下角的元素被设置为 0
        scores[:, :, :, -seq_len:] += torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=scores.device),
            diagonal=1,
        )

        # 外部掩码（屏蔽 [PAD] 或逻辑不关块）
        # attention_mask 的形状为 [bsz, total_len]
        if attention_mask is not None:
            extended_attention_mask = attention_mask.unsqueeeze(1).unsqueeze(2)  # [bsz, 1, 1, total_len]

            # 原掩码 1 表示注意，0 表示不注意
            extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
            scores += extended_attention_mask

        # softmax 归一化得到注意力权重
        # type_as(xq) 确保 scores 的 dtype 与 xq 相同，避免数值不稳定
        # scores 的形状为 [bsz, heads, seq_len, total_len]
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        scores = self.attn_dropout(scores)

        # xv 的形状为 [bsz, n_k_v_heads, total_len, head_dim]
        output = scores @ xv  # 此时 output 的形状为 [bsz, n_heads, seq_len, head_dim]

    # 恢复形状 + 残差 dropout
    # 先 reshape 为 [bsz, seq_len, n_heads * head_dim] 找回形状，再通过线性层投影回 hidden_size 维度
    output = output.transpose(1, 2).reshape(bsz, seq_len, -1)  # [bsz, seq_len, hidden_size]
    output = self.resid_dropout(self.o_proj(output))

    # output 输出用于后续的残差连接
    # 累积的 kv 缓存
    return output, past_kv
```

## 相关笔记

- **总览**：[[10-LLM-大语言模型/01-Architecture-模型架构/MiniMind架构图解析|MiniMind架构图解析]] — 架构图说明了 GQA 在完整模型中的位置。
- **前置**：[[10-LLM-大语言模型/01-Architecture-模型架构/RoPE位置编码与YaRN|RoPE位置编码与YaRN]] — 注意力中的 Query 和 Key 是 RoPE 旋转变换的直接对象。
- **系统影响**：[[20-AI-Infra-AI基础设施/01-Roadmaps-学习路线/路线|vLLM／SGLang 路线]] — GQA 通过减少 KV Head 数影响 KV Cache 和推理显存。
