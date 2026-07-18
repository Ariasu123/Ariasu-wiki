---
date: 2026-04-15
---
![](../_assets/Architecture-%E6%A8%A1%E5%9E%8B%E6%9E%B6%E6%9E%84/%E6%9E%B6%E6%9E%84%E5%9B%BE.png)
### 1. 整体流程

关联阅读：可结合 [[10-LLM-大语言模型/01-Architecture-模型架构/Attention 与 GQA|Attention 与 GQA]] 理解；GQA 是 MiniMind Transformer 层的核心注意力模块。


这是模型处理信息的全过程：

- **Tokenizer Encoder（分词编码）**：把我们人类看的“hello”转换成机器认识的数字编号（Input IDs）。
- **Input Embedding（嵌入层）**：把数字编号变成一串高维向量（数值列表），赋予单词语义特征
- **Transformer Layer k（核心加工层）**：这是模型的大脑，图中显示它会重复 $K$ 次（循环叠加），每一层都会让模型对文本的理解更深一层。
- **RMSNorm & Linear & SoftMax**：最后加工环节，将深层特征转换成概率，决定下一个词选什么的概率最大。
- **Tokenizer Decoder（分词解码）**：把概率最高的数字变回人类语言，输出“world”。
---
### 2. 单层结构

这是每一个 Transformer 层内部的“两步走”战略：

- **GQA (Grouped Query Attention)**：第一步，负责“注意力”。让模型在看当前词时，能注意到上下文里哪些词是重要的。
- **FFN (Feed Forward Network)**：第二步，负责“知识记忆”。对注意力提取的信息进行非线性变换，提取更抽象的特征。
- **旁边的“+”号和箭头**：这叫**残差连接（Residual Connection）**。简单说就是“保留原文，加上新学到的东西”，防止网络太深导致信息丢失。
---
### 3. 核心组件细节（右侧 a & b）

#### (a) GQA - 分组查询注意力（核心创新）

这是现代大模型节省显存、提高速度的关键：

- **Q, K, V**：查询(Query)、键(Key)、值(Value)。就像在图书馆（K, V）用搜索词（Q）找书。
- **RoPE (Rotary Positional Embedding)**：旋转位置编码。它告诉模型每个词在句子里的**相对位置**，没有它模型就分不清“人咬狗”和“狗咬人”。 
- **mask**：掩码。确保模型在预测下一个词时，看不见后面的“标准答案”。
- **SoftMax**：计算权重，决定要把多少注意力放在哪个词上。
#### (b) FFN - 前馈神经网络（SwiGLU结构）

MiniMind 使用了高效的 **SwiGLU** 激活函数：

- **RMSNorm**：在进入加工前先做归一化，让数据分布更稳定。
- **SiLU**：一种激活函数。你可以理解为一种“开关”，决定哪些信息可以通过。
- **三个 Linear 层**：通过矩阵运算进行高维空间的映射。这里是模型存储“知识”的地方。
------------------

在 MiniMind（以及所有类 Llama 架构）中，**Linear 层（线性变换层）** 本质上是矩阵乘法 $y = xW$

### 1. 核心层入口处的 Linear（位于 GQA 模块内部）

在 **(a) GQA** 模块最下方的那个蓝色 `Linear` 框：

- **作用：生成 Q、K、V 向量**。
    
- **原理**：输入的词向量（Embedding）是原始的，它并不直接包含“谁是谁的宾语”这种关系。通过这个 Linear 层，模型将原始向量投影到三个不同的子空间：
    - **Q (Query)**：我想找什么信息？
    - **K (Key)**：我有多少信息可以提供？
    - **V (Value)**：我具体包含的内容是什么？
    
     在Linear层里和QKV矩阵计算，获取对应向量，其中Q和K进行RoPE位置编码，然后点积，掩码处理后softmax，获得一个“相关性矩阵”，然后输入V点积，获得具体的上下文侧重信息
### 2. 注意力机制后的 Linear（位于 GQA 模块上方）

在 **(a) GQA** 模块顶部的那个蓝色 `Linear` 框：

- **作用：特征融合与统一输出**。
    
- **原理**：经过注意力计算后，模型得到了很多分散的、加权后的上下文信息。这个 Linear 层的作用是把多头注意力整合，并映射回模型原本的维度，以便传给下一层。    

---
### 3. FFN 模块中的三个 Linear（位于 b 模块中）

这是模型存储“知识”的地方，图中 **(b) FFN** 展示了典型的 **SwiGLU** 结构，这里有三条线涉及 Linear：

- **左侧和右侧并行的两个 Linear**：
    - **作用：空间扩张与门控控制**。
    - **原理**：这两个 Linear 层通常会将向量的维度升得很高（比如从 512 升到 1408）。
        - 右边的经过 **SiLU** 激活函数变成一个“门”；
        - 左边的提供原始特征；
        - 两者相乘，实现**信息的筛选**。
- **顶部的 Linear**
    - **作用：维度还原**。
    - **原理**：将刚才升高的维度重新压回到模型的标准维度。
---
### 4. 模型最顶部的 Linear（输出层）

在最左侧大图的 **RMSNorm** 之上、**SoftMax** 之下：

- **作用：预测词表概率**。
    
- **原理**：这是最后一步。它将模型处理了半天的深层特征，映射到整个**词表**的大小（比如 MiniMind 词表如果有 64000 个词，那输出维度就是 64000）。
------

架构设计：

### 1. 为什么用 GQA（分组查询注意力）而不是传统的 Attention？

传统的 Attention（MHA）每个“头”都有自己的 K 和 V，非常占显存。

- **设计目的：** **极致的推理速度和显存节省。**
- **原理：** GQA 让多个 Q（查询）共享同一组 K 和 V。这大大减少了在生成文字时需要缓存的数据量（KV Cache）。

---
### 2. 为什么选择 SwiGLU（FFN 里的那种设计）？

你在图中看到 FFN 模块分成了两条路，一条路走 SiLU 激活函数，然后两路相乘。

- **设计目的：** **提升模型的“理解深度”和“非线性表达能力”。**
- **原理：** 传统的激活函数（如 ReLU）太死板，要么开要么关。SwiGLU 这种“门控”机制允许模型更精细地控制哪些信息该留下，哪些该丢弃。

| 架构图组件 | 对应类 / 成员 | 作用说明 |
| --- | --- | --- |
| **Input Embedding** | `MindModel.embed_tokens` | 将 token id 映射成初始向量表示，相当于“查表取词向量”。 |
| **Layer k（堆叠 K 次）** | `MindModel.layers` | 按层堆叠 Transformer Block；在推理阶段也负责串起整条计算链路和 KV Cache。 |
| **RMSNorm** | `RmsNorm` | 稳定特征分布，减少深层网络中的数值爆炸或训练不稳定。 |
| **(a) GQA 模块** | `Attention`<br>`repeat_kv`<br>`apply_rotary_pos_emb`<br>`precompute_freqs` | 完成分组查询注意力、RoPE 位置编码和 KV 复用，是注意力计算的核心部分。 |
| **(b) FFN 模块** | `FeedForward`<br>`MOEFeedForward` | 通过 SwiGLU / FFN 做非线性变换，提取更高层特征并增强表达能力。 |
| **$\oplus$（残差连接）** | `TransformerBlock` 内部的 `+` | 保留输入并叠加子层输出，帮助梯度传播，缓解深层网络退化。 |
| **顶层 Linear & SoftMax** | `MindForCausalLM` | 将隐藏状态映射回词表 logits，并通过 `softmax / cross_entropy` 完成训练或生成时的选词。 |

## 相关笔记

- **组成**：[[10-LLM-大语言模型/01-Architecture-模型架构/Attention 与 GQA|Attention 与 GQA]] — GQA 是 MiniMind Transformer 层的核心注意力模块。
- **组成**：[[10-LLM-大语言模型/01-Architecture-模型架构/RMSNorm和LayerNorm|RMSNorm和LayerNorm]] — RMSNorm 是模型层和输出层稳定训练的关键组件。
- **组成**：[[10-LLM-大语言模型/01-Architecture-模型架构/RoPE位置编码与YaRN|RoPE位置编码与YaRN]] — RoPE 为注意力计算注入位置信息。
- **扩展**：[[10-LLM-大语言模型/01-Architecture-模型架构/MoE学习笔记|MoE学习笔记]] — MoE 是对稠密 FFN 的稀疏专家化扩展。
- **实现**：[[10-LLM-大语言模型/01-Architecture-模型架构/拼接模型并封装为Hugging Face格式|拼接模型并封装为Hugging Face格式]] — 模型组装后需要封装成可保存和加载的 Hugging Face 形式。
