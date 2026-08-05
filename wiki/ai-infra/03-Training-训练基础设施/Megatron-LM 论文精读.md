---
title: Megatron-LM 论文精读
tags: [megatron-lm, 张量并行, 模型并行, 分布式训练, ai-infra]
created: 2026-08-05
updated: 2026-08-05
status: draft
---

> 信源：Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism，Mohammad Shoeybi 等（NVIDIA），NeurIPS 2019，[arXiv:1909.08053](https://arxiv.org/abs/1909.08053)，[开源代码](https://github.com/NVIDIA/Megatron-LM)。本页为该论文的网状知识提炼，非全文转载。

**Megatron-LM** 是 NVIDIA 提出的**层内模型并行（intra-layer model parallelism，即张量并行）**训练方案：不依赖任何新编译器或框架改造，只在一个原生 PyTorch transformer 实现里插入几个 all-reduce 通信原语，就能把 GPT-2/BERT 类模型训练到 83 亿参数（512 张 V100），并拿到当时多项 SOTA。这篇论文也是训练基础设施领域"张量并行"范式的奠基之作。

---

## 一、要解决什么问题：显存墙

大模型训练最大的物理约束是**单卡显存**：

- 权重本身占内存，而 **ADAM 优化器还要为每个参数额外保存动量与方差状态**，大幅压缩了单卡能承载的模型规模。
- 激活检查点（activation checkpointing，反向时重算激活）只能缓解，不能根治——**模型必须完整放进一张卡**才能训练。
- 已有方案（GPipe 流水线并行、Mesh-TensorFlow）需要重写模型、依赖仍在开发中的自定义编译器/框架，工程成本高。

结论：需要一种**简单、高效、不依赖编译器**的方式把模型切分到多卡。详见 [AI Infra 领域概览](../02-Foundations-%E5%9F%BA%E7%A1%80%E6%A6%82%E5%BF%B5/AI%20Infra%20%E9%A2%86%E5%9F%9F%E6%A6%82%E8%A7%88.md) 中的"显存墙"与"训练 infra 工程难点"。

## 二、核心思想：层内模型并行（张量并行）

关键洞察：**transformer 层内的矩阵运算天然可以按张量维度切分**，且切分后的局部计算不需要立即通信。方案要点：

- 对 MLP 块和自注意力块的 GEMM 按列/行切分，使大多数计算在单卡内完成。
- 只在**层与层之间**插入少量 all-reduce 完成张量聚合，每个 transformer 层前向 2 次 + 反向 2 次，共 4 次通信。
- 与流水线并行（PP）**正交且互补**，可叠加使用。

通用切分策略的完整分析见 [张量并行（模型并行）](%E5%BC%A0%E9%87%8F%E5%B9%B6%E8%A1%8C%EF%BC%88%E6%A8%A1%E5%9E%8B%E5%B9%B6%E8%A1%8C%EF%BC%89.md)。

### 2.1 MLP 块的切分

MLP 块是一个 `GEMM → GeLU → GEMM` 结构，第一个 GEMM 是 $Y = \text{GeLU}(XA)$：

- **按行切分 $A$**（$A = [A_1, A_2]^\top$）会使每卡只算出一部分和，需要**在 GeLU 之前**做一次求和同步。由于 GeLU 是非线性函数，$\text{GeLU}(X_1A_1 + X_2A_2) \neq \text{GeLU}(X_1A_1) + \text{GeLU}(X_2A_2)$，无法先算后合——引入一个同步点。
- **按列切分 $A$**（$A = [A_1, A_2]$）则每卡独立算出 $[Y_1, Y_2] = [\text{GeLU}(XA_1), \text{GeLU}(XA_2)]$，**GeLU 可独立应用，消除一个同步点**。

因此第一个 GEMM 采用**列并行（column parallel）**，第二个 GEMM 采用**行并行（row parallel）**，直接消费 GeLU 输出；第二个 GEMM 的输出再在 GPU 间做一次 all-reduce，进入 dropout/残差。两个 GEMM 融合为一组，前向只需一次 all-reduce。

### 2.2 f / g 通信算子

切分后的"连接点"用一对共轭算子表达：

- **$f$ 算子**：前向恒等（identity），反向做 all-reduce（梯度聚合）。
- **$g$ 算子**：前向做 all-reduce（激活聚合），反向恒等。

PyTorch 里只需几行代码：

```python
class f(torch.autograd.Function):
    def forward(ctx, x):
        return x
    def backward(ctx, gradient):
        all_reduce(gradient)
        return gradient
```

### 2.3 自注意力块的并行

利用多头注意力的天然并行性：将 **Q、K、V 的投影 GEMM 按列切分**，使每个 attention head 的矩阵乘法完全在单卡内完成，**无需任何即时通信**；随后的输出线性层按行切分，直接消费并行 attention 的输出。与 MLP 一样，每层前向/反向各只需 2 次 all-reduce。

### 2.4 Embedding 并行与 Loss 融合

输出 embedding 尺寸为 hidden-size × vocab-size（词表常达数万，GPT-2 为 50,257），值得并行：

- 输入 embedding 按**词表维度**列切分，每卡持有一部分表，嵌入后需要一次 all-reduce（$g$ 算子）。
- 输出 embedding（与输入共享权重）算出各卡局部 logits 后，朴素做法是 all-gather 全部 logits，通信量为 $b \times s \times v$（batch × 序列长 × 词表），过大。
- **优化：把并行 logits 与交叉熵 loss 融合**，各卡先算局部交叉熵再 all-reduce 标量 loss，通信量降到 $b \times s$。这是论文强调的显著通信缩减。

### 2.5 复制计算策略

对于 dropout、LayerNorm、残差连接这类**计算轻、张量小**的操作，不采用"单卡算完再广播"（广播也有通信成本），而是**在各卡上保留重复副本重复计算**。每个 model parallel worker 独立优化自己持有的参数，因为所有值要么本地、要么有副本，**不需要通信参数更新**。

## 三、混合并行：张量并行 × 数据并行

张量并行（模型内并行）与数据并行（跨样本并行）正交，可叠加：

- 8 张 GPU 组成一个**模型并行组**（同机内，8-way TP），组内做 all-reduce 聚合张量。
- 多个模型并行组之间，按**组内相同位置**组成数据并行组，组间做梯度 all-reduce（64-way DP）。
- 总 GPU 数 = 模型并行组大小 × 数据并行组数：8 × 64 = **512 张 V100**。

数据并行的梯度 all-reduce 引入额外通信，扩展效率略降，但依然可观。

## 四、实验结果

### 4.1 扩展效率

- 基线：1.2B 参数单卡（V100 32GB，DGX-2H）训练中稳定 **39 TeraFLOPS**，约为单卡理论峰值的 **30%**——这是一个很强的基线。
- 8.3B / 8-way TP：**77%** 线性扩展效率。
- 8.3B / 512 GPU（TP+DP）：应用全程 **15.1 PetaFLOPs**，**76%** 扩展效率。
- 强扩展（固定 1.2B 模型 + 固定 batch=8，只加卡）：1→2→4→8 卡加速比 1.0→1.64→2.34→2.98，超过 2 卡后收益递减（单卡计算变小，带宽与通信开销占主导）。
- 注意力头数量影响：固定 8.3B/8-way，head 数从 16→32，扩展效率 82%→77%（head 越多单 GEMM 越小、softmax 元素越多）。

### 4.2 GPT-2 系列（左到右生成式）

| 参数 | 层数 | Hidden | 训练 GPU | WikiText103 困惑度↓ | LAMBADA 准确率↑ |
|------|------|--------|----------|---------------------|------------------|
| 355M | 24 | 1024 | 64 | 19.31 | 45.18% |
| 2.5B | 54 | 1920 | 128 | 12.76 | 61.73% |
| 8.3B | 72 | 3072 | 512 | **10.81** | **66.51%** |
| 当时 SOTA | — | — | — | 15.79 | 63.24% |

- 8.3B 模型在 512 GPU 上每个 epoch（68,507 迭代）约 2 天。
- 验证困惑度随规模单调下降，8.3B 达到 9.27。

### 4.3 BERT 系列（双向编码式）与 LayerNorm 重排

**重要架构发现**：BERT 类模型在超过 BERT-Large（336M）后性能退化。论文通过**重排 LayerNorm 与残差连接的顺序**（将 LayerNorm 移到子层之前，即 Pre-LN 风格）消除了训练不稳定、降低了训练 loss——这是**首次报道该改动能支撑训练更大 BERT**。此发现与 [RMSNorm和LayerNorm](../../llm/01-Architecture-%E6%A8%A1%E5%9E%8B%E6%9E%B6%E6%9E%84/RMSNorm%E5%92%8CLayerNorm.md) 中"归一化稳定梯度"的机制互补：不仅"怎么归一化"重要，**"归一化放在哪"**同样关键。

- 336M（≈BERT-Large）、1.3B、3.9B 三档，held-out 验证困惑度 1.58 → 1.30 → 1.16 单调下降。
- 下游任务（MNLI、QQP、SQuAD、RACE）全部随规模提升；3.9B 在 RACE 测试集达 90.9%（当时 SOTA 89.4%），单模型与 5 路 ensemble 均为 SOTA。

### 4.4 训练细节

- **数据**：Wikipedia、CC-Stories、RealNews、OpenWebText（BERT 另加 BooksCorpus），过滤 <128 token 文档，用**局部敏感哈希（LSH）按 Jaccard 相似度 >0.7 去重**，最终 174GB 去重文本。
- **优化**：混合精度 + 动态 loss scaling（利用 V100 Tensor Core）、权重初始化 $\mathcal{N}(0, 0.02)$、AdamW（权重衰减 0.01）、全局梯度范数裁剪 0.1、dropout 0.1、逐 transformer 层激活检查点。
- GPT-2：序列 1024、batch 512、300k 迭代，学习率 1.5e-4 经 3k warmup 后单周期余弦衰减至 1e-5。

## 五、影响与后续

- 微软与 NVIDIA 合作训练的 **Turing-NLG 17B** 模型正是基于 Megatron，验证了规模继续扩大时准确率进一步提升。
- 论文展望的"混合 intra-layer（张量）+ inter-layer（流水线）+ inter-node"方案成为后续 Megatron 系列的核心路线。
- 张量并行作为范式被 DeepSpeed、ColossalAI 及主流训练框架沿用，是分布式训练的三大并行基石之一（详见 [张量并行（模型并行）](%E5%BC%A0%E9%87%8F%E5%B9%B6%E8%A1%8C%EF%BC%88%E6%A8%A1%E5%9E%8B%E5%B9%B6%E8%A1%8C%EF%BC%89.md)）。

## 相关笔记

- **核心概念**：[张量并行（模型并行）](%E5%BC%A0%E9%87%8F%E5%B9%B6%E8%A1%8C%EF%BC%88%E6%A8%A1%E5%9E%8B%E5%B9%B6%E8%A1%8C%EF%BC%89.md) — column/row parallel 切分、f/g 算子、通信开销的通用分析。
- **领域定位**：[AI Infra 领域概览](../02-Foundations-%E5%9F%BA%E7%A1%80%E6%A6%82%E5%BF%B5/AI%20Infra%20%E9%A2%86%E5%9F%9F%E6%A6%82%E8%A7%88.md) — Megatron-LM 作为训练 infra 代表框架、显存墙与并行算法总览。
- **归一化机制**：[RMSNorm和LayerNorm](../../llm/01-Architecture-%E6%A8%A1%E5%9E%8B%E6%9E%B6%E6%9E%84/RMSNorm%E5%92%8CLayerNorm.md) — 归一化的梯度稳定原理，与 Pre-LN 位置发现互补。
- **训练管线**：[LLM预训练底层机制](../../llm/02-Pretraining-%E9%A2%84%E8%AE%AD%E7%BB%83/LLM%E9%A2%84%E8%AE%AD%E7%BB%83%E5%BA%95%E5%B1%82%E6%9C%BA%E5%88%B6.md) — 预训练的数据管线与训练引擎视角。
- **MoE 关联**：[MoE 学习笔记](../../llm/01-Architecture-%E6%A8%A1%E5%9E%8B%E6%9E%B6%E6%9E%84/MoE%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0.md) — 专家并行（EP）与张量并行的分工关系。
