# Attention 优化技术综述

> 覆盖 FlashAttention 家族（v1–v4）、PagedAttention、稀疏注意力（NSA/MoBA/DSA 等）与分布式长上下文方案（Ring/Ulysses）四条主线。写作原则：每个技术都回答四个问题——**为什么出现？解决什么瓶颈？怎么做的？和其他技术什么关系？** 所有关键数字均标注出处；论文口径与社区说法有出入时会明确区分。
>
> 相关页面：[KV Cache 内存管理（PagedAttention）](KV%20Cache%20%E5%86%85%E5%AD%98%E7%AE%A1%E7%90%86%EF%BC%88PagedAttention%EF%BC%89.md)、[vLLM 论文精读](vLLM%20%E8%AE%BA%E6%96%87%E7%B2%BE%E8%AF%BB.md)、[Attention 与 GQA](../../LLM/01-Architecture-%E6%A8%A1%E5%9E%8B%E6%9E%B6%E6%9E%84/Attention%20%E4%B8%8E%20GQA.md)

---

## 0. 一张图看懂全局（先给结论）

标准 Attention 的复杂度是 O(N²)（N 为序列长度），但"慢"和"贵"其实来自**四个不同的瓶颈维度**，每类技术只打其中一个点，彼此**正交、可叠加**：

| 瓶颈维度 | 代表技术 | 一句话概括 |
|---|---|---|
| **单机 kernel 的显存 IO** | FlashAttention v1→v4 | 不在 HBM 里实例化 N×N 注意力矩阵，分块在 SRAM 里算 |
| **KV cache 的显存管理** | PagedAttention (vLLM) | 像 OS 虚拟内存一样分页管理 KV cache，消灭碎片 |
| **注意力计算量本身** | 稀疏注意力（NSA/MoBA/DSA 等） | 每个 query 只看一小部分 key，把 O(N²) 压到近线性 |
| **单机放不下的超长序列** | Ring Attention / Ulysses | 把序列切到多卡，通信与计算重叠 |

一个现代推理引擎（如 vLLM/SGLang）是这些技术的**叠加态**：PagedAttention 管显存分配，FlashAttention 算每个 kernel，稀疏注意力减少每个 query 要读的 KV，多卡时用 Ring/Ulysses 扩展序列维度。

---

## 1. 背景：Attention 到底慢在哪？

标准 scaled dot-product attention：

```
S = Q @ K^T / sqrt(d)      # (N, N) 打分矩阵
P = softmax(S)             # (N, N) 概率矩阵
O = P @ V                  # (N, d) 输出
```

教科书会告诉你瓶颈是 O(N²) 的**计算量**（FLOPs）。但 FlashAttention 论文（arXiv:2205.14135）开篇就纠正了这个直觉：

> **真正的瓶颈往往不是 FLOPs，而是显存 IO（HBM 读写次数）。**

GPU 的内存是分层的（以 A100 为例）：

- **SRAM**（片上共享内存/寄存器）：每 SM 约 192KB，带宽 ~19 TB/s，快但小
- **HBM**（显存）：40–80GB，带宽 ~1.5–2 TB/s，大但慢一个数量级

标准实现要把 (N, N) 的 S 和 P 矩阵**显式写回 HBM、再读回来**做下一步。N=8K 时这个矩阵就是 128M 个元素——大量时间花在"搬运"而非"计算"上。这类 kernel 叫 **memory-bound**（对比：大矩阵乘是 compute-bound）。

FlashAttention 的核心思想由此而来：**IO-aware**——设计算法时显式优化 HBM↔SRAM 的读写次数，而不只盯着 FLOPs。FA1 论文还给出了 IO 复杂度分析：HBM 访问从标准实现的 Θ(Nd + N²) 降到 Θ(N²d²/M)（M 为 SRAM 大小），并证明这在一定 SRAM 范围内是最优的。

---

## 2. 基石：从 Safe Softmax 到 Online Softmax

理解 FlashAttention 的钥匙是 online softmax。这条线索（3-pass → 2-pass → 1-pass）也是中文社区标杆文章（知乎 DefTruth《从 Online-Softmax 到 FlashAttention V1/V2/V3》）的叙事主线。

### 2.1 Safe Softmax：3-pass

直接算 softmax 会数值溢出（exp(大数)），所以要先减最大值：

```
# pass 1: 求最大值
m = max(x)
# pass 2: 求指数和
l = sum(exp(x - m))
# pass 3: 归一化
p = exp(x - m) / l
```

问题：x 要被**从 HBM 读 3 遍**。

### 2.2 Online Softmax：2-pass（Milakov & Gimelshein, NVIDIA 2018）

观察：max 和 sum 可以**一趟流式算完**——维护 running max `m` 和 running sum `l`，每来一个新元素就更新，并用 rescale 修正旧的累积值：

```
m_new = max(m_old, x_i)
l_new = l_old * exp(m_old - m_new) + exp(x_i - m_new)
```

把所有 x 扫完后，最后的 `exp(x_i - m) / l` 就是精确结果。**数学上与 safe softmax 完全等价，不是近似。**

### 2.3 通向 FlashAttention：1-pass

Attention 里 softmax 后面还跟着 `P @ V`。把"更新 (m, l)"和"累加 O += exp(S_ij - m) @ V_j"融进**同一趟分块循环**，就得到了 FlashAttention 的核心循环：

```
# FlashAttention 核心循环（伪代码，FA2 风格：外循环 Q，内循环 K/V）
for each Q_block:                    # 外循环：Q 分块（FA2 的改法）
    O, m, l = 0, -inf, 0             # 初始化输出累加器和 softmax 统计量
    for each (K_block, V_block):     # 内循环：流式扫 K/V
        S = Q_block @ K_block^T / sqrt(d)       # 局部打分块（在 SRAM 里）
        m_new = max(m, rowmax(S))
        l = l * exp(m - m_new) + rowsum(exp(S - m_new))
        O = O * exp(m - m_new) + exp(S - m_new) @ V_block   # rescale 旧结果再累加
        m = m_new
    O = O / l                        # 循环结束统一归一化（FA2 的延迟 rescale）
```

整个过程中 (N, N) 的 S、P 从未在 HBM 中完整存在过——**额外显存从 O(N²) 降到 O(N)**。

---

## 3. FlashAttention 家族：v1 → v2 → v3 → v4

演进主线一句话：**算法（省 IO）→ 调度（提并行）→ 硬件适配（Hopper 异步/FP8）→ 新硬件协同（Blackwell）**。

### 3.1 FlashAttention v1（2022，NeurIPS）

论文：Dao, Fu, Ermon, Rudra, Ré, *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*（arXiv:2205.14135）

**三大技术**：

1. **Tiling（分块）**：Q/K/V 切成小块，每次只把一小块载入 SRAM 计算，不实例化完整注意力矩阵；
2. **Online Softmax**：分块递推 softmax 统计量（见 §2），结果精确；
3. **Recomputation（重计算）**：反向传播时不存 N×N 注意力矩阵，只存输出 O 和统计量 (m, l)，反向时在片上重算——以算力换显存，而算力恰恰是富余的。

**性能（论文口径）**：GPT-2（seq 1K）训练快 3×；BERT-large 端到端快 15%；Tri Dao 演讲口径为相比优化 baseline 墙钟 2–4×、显存省 10–20×。

**留下的问题**：FA1 的循环是**外循环 K/V、内循环 Q**（论文 Algorithm 1）——每个 K/V 块要扫所有 Q 块，Q 和 O 反复从 HBM 重载，且并行度受限于 batch×heads。结果：FA1 在 A100 上只达到理论峰值 FLOPs 的 **25–40%**（FA2 论文摘要口径）。

### 3.2 FlashAttention v2（2023，ICLR 2024）

论文：Tri Dao, *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*（arXiv:2307.08691）

FA1 是算法突破，FA2 是"把 GPU 榨干净"的工程重构，三个改进：

1. **交换循环顺序**：外循环改为 **Q**（每个线程块负责一个 Q 块，内循环流式扫 K/V）。不同 Q 块的计算完全独立 → 可以在序列维度上大量并行，batch 小/序列长时也能占满 GPU；Q 块只加载一次，rescale 次数大减。
2. **削减 non-matmul FLOPs**：softmax 的缩放/除法等非矩阵乘运算跑在 CUDA core 上，远慢于 Tensor Core。FA2 延迟输出 rescaling（循环结束才统一除以 l），把尽可能多的运算留给 Tensor Core。
3. **Warp 级工作重划分**：从 FA1 的 split-K（各 warp 切 K/V，需经 shared memory 通信归约）改为 **split-Q**（每个 warp 处理不同 Q 行块、共享 K/V），消除 warp 间通信。

**性能（论文口径）**：比 FA1 约 **2×**；A100 上达理论峰值 **50–73%**，端到端训练 GPT 类模型达 225 TFLOPs/s per A100（72% MFU），接近 GEMM 的效率。

### 3.3 FlashAttention v3（2024，NeurIPS 2024）

论文：Shah 等（Colfax/Meta/NVIDIA/Princeton 等），*FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision*（arXiv:2407.08608）

FA2 在 H100 上只用了 35% 的算力——因为 Hopper 架构的新特性没被利用。FA3 专门为 H100 重写调度：

1. **Warp Specialization（生产者-消费者）**：一部分 warp 专职用 TMA（Hopper 的异步拷贝单元）搬数据，另一部分专职算 GEMM+softmax，配合环形 SMEM buffer 让**数据搬运和计算真正重叠**。
2. **GEMM-Softmax 指令级流水**：动机是一个悬殊的数字——H100 的 FP16 matmul 有 989 TFLOPS，而特殊函数单元（算 exp）只有 ~3.9 TFLOPS（差 256 倍），softmax 不藏起来会吃掉一半时间。具体做法包括 warpgroup 间 **pingpong 调度**（两个 consumer warpgroup 轮流做 GEMM 和 softmax）和 warpgroup 内 GEMM/softmax 交错。
3. **FP8 低精度**：block quantization（分块缩放）+ incoherent processing（用随机 Hadamard 变换把量化误差打散到各维度），FP8 误差比 baseline FP8 attention 低 2.6×。

**性能（arXiv 摘要口径）**：H100 上比 FA2 快 **1.5–2.0×**；FP16 最高 **740 TFLOPs/s（75% 利用率）**；**FP8 接近 1.2 PFLOPs/s**。
（注：社区有文章转述 NeurIPS 最终版的 BF16 840 TFLOPs/85% 数字，与 arXiv 摘要口径略有出入，引用时注意版本。）

### 3.4 FlashAttention v4（2026，面向 Blackwell）

论文：Zadouri, Hoehnerbach, Shah 等，*FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling*（arXiv:2603.05451，2026-03）。Tri Dao 在 Hot Chips 2025 已展示初步结果；Dao-AILab 官方仓库已发布（CuTe-DSL 实现，`pip install flash-attn-4`）。

**动机**：Blackwell（B200/GB200）是**非对称扩展**——Tensor Core 吞吐翻倍，但 SMEM 带宽、exp 单元等其他单元扩展慢甚至不变，瓶颈位置变了。

**关键技术**：

1. 重新设计的流水线：利用完全异步的 MMA（tcgen05.mma）和更大 tile；
2. **软件模拟指数**：exp 不用慢速 SFU，改用 FMA 单元做多项式逼近；**条件 softmax rescaling**——只在 max 变化大到威胁数值稳定时才 rescale（Hot Chips 报告称 rescale 次数减少约 10×）；
3. 利用 **TMEM**（Blackwell 每 SM 256KB 的张量内存，直接存 MMA 中间结果）和 2-CTA MMA 模式，减少 SMEM 流量；
4. 完全用 **CuTe-DSL**（Python 嵌入式 DSL）实现，编译时间比 C++ 模板快 20–30×。

**性能（论文口径）**：B200 上 BF16 最高 **1613 TFLOPs/s（71% 利用率）**，比 cuDNN 9.13 快至 1.3×、比 Triton 快至 2.7×。

### 3.5 四代对比表

| 版本 | 年份 | 核心目标 | 关键技术 | 硬件 | 论文口径性能 |
|---|---|---|---|---|---|
| **FA1** | 2022 | 省 HBM IO | tiling + online softmax + 重计算 | 通用（A100 等） | GPT-2 训练 3×；显存 O(N²)→O(N)；A100 利用率 25–40% |
| **FA2** | 2023 | 提并行度/利用率 | 外循环换 Q；削减 non-matmul FLOPs；split-Q warps | Ampere 优化 | 比 FA1 约 2×；A100 峰值 50–73% |
| **FA3** | 2024 | 吃满 Hopper | warp specialization；GEMM-softmax 流水；FP8 | H100 | 比 FA2 1.5–2×；FP16 740 TFLOPs(75%)；FP8 ~1.2 PFLOPs |
| **FA4** | 2026 | 适配 Blackwell 非对称扩展 | 软件模拟 exp；条件 rescale；TMEM/2-CTA；CuTe-DSL | B200/GB200（兼容 Hopper） | BF16 1613 TFLOPs(71%)；比 cuDNN 快至 1.3× |

### 3.6 番外：FlashDecoding 与 FlashDecoding++

FlashAttention 解决的是训练/prefill（Q 很长）。**Decode 阶段 Q 只有 1 个 token**，batch×heads 维度并行度不够，GPU 占不满——瓶颈又变了。

- **FlashDecoding**（2023，无独立论文，官方出处是 PyTorch/CRFM 博客）：新增 **K/V 序列长度维度的并行（split-K）**——把 K/V 切块，每块并行算局部注意力并记录 log-sum-exp，最后一个 reduction kernel 按 log-sum-exp 加权合并。长序列生成最高 8× 加速。vLLM 的 PagedAttention V2 kernel、TensorRT-LLM 的 multi_block_mode 都是同一思路。
- **FlashDecoding++**（arXiv:2311.01282，清华/无问芯穹等，与 Tri Dao 团队无关）：统一最大值的异步 softmax（各 partial 用预设 max，避免同步）+ flat GEMM 双缓冲 + 硬件自适应数据流；相比 FlashDecoding 平均再快 1.37×。

---

## 4. PagedAttention：KV cache 的"虚拟内存"

论文：Kwon 等，*Efficient Memory Management for Large Language Model Serving with PagedAttention*（arXiv:2309.06180，SOSP 2023）。系统：vLLM。更详细的拆解见站内页面 [KV Cache 内存管理（PagedAttention）](KV%20Cache%20%E5%86%85%E5%AD%98%E7%AE%A1%E7%90%86%EF%BC%88PagedAttention%EF%BC%89.md) 与 [vLLM 论文精读](vLLM%20%E8%AE%BA%E6%96%87%E7%B2%BE%E8%AF%BB.md)。

### 4.1 它解决的问题和 FlashAttention 完全不同

FA 优化的是**单个 attention kernel 内部**的 IO；PagedAttention 优化的是**推理服务层面 KV cache 的显存分配**。

背景：decode 时每生成一个 token 都要用到全部历史 KV，所以 KV cache 必须常驻显存（量级感受：OPT-13B 单 token 的 KV 就是 800KB，2048 token 的请求要 1.6GB）。传统做法是为每个请求**按最大可能长度预分配连续显存**，vLLM 论文实测（Fig. 2）：现有系统只有 **20.4%–38.2%** 的 KV cache 显存真正在存 token 状态——即浪费 60%–80%（"60–80%"这个整数区间出自 vLLM 官方博客，论文原始数据是利用率口径）。

浪费由**三类**构成（注意：社区文章常只列两类，论文是三类）：

1. **Reserved 预留**：为未来 token 预留的槽位，请求活着就一直占着；
2. **内部碎片**：按最大长度超配，请求提前结束时差额纯浪费；
3. **外部碎片**：不同请求分配大小不一，释放后留下无法复用的空洞。

### 4.2 核心机制：照抄操作系统的分页

PagedAttention 把 KV cache 类比成 OS 的虚拟内存（论文明确引用 1962 年 Atlas 分页论文）：

| OS 虚拟内存 | PagedAttention |
|---|---|
| 页（page） | KV block（默认 **16 个 token**，论文 §7.2 实测选定） |
| 字节 | token |
| 进程 | 请求 |
| 页表 | block table（逻辑块 → 物理块映射 + 已填充数） |
| 按需分配物理页 | 只有当前块填满才分配新物理块 |

效果：**逻辑上连续、物理上离散**。每个请求的浪费被限制在**最后一个未填满的 block 之内**——论文表述为 "near-zero waste"，官方博客给出的数字是 **<4%**。

### 4.3 杀手锏：Copy-on-Write 共享

物理块带引用计数，多个序列可以映射到同一物理块；某序列要写入引用计数 >1 的块时，按块粒度拷贝（类比 OS 的 fork）。这让以下场景大幅省显存：

- **Parallel sampling**（同一 prompt 采多个答案）：prompt 的物理块全部共享；
- **Beam search**：不同 beam 除 prompt 外还能共享中间块，共享模式随解码动态变化；
- **Shared prefix**（共享系统提示词）：预存前缀的 KV 物理块，新请求直接映射过去。

论文实测省显存：beam search 场景省 **37.6%–66.3%**（Alpaca/ShareGPT trace），对应吞吐提升最高 2.2×。

### 4.4 抢占与恢复：swapping vs recomputation

显存不够时按 FCFS 抢占最晚到达的请求（all-or-nothing 驱逐，beam 等同组序列一起调度），恢复有两条路：

- **Swapping**：块拷到 CPU RAM，回头再拷回来；
- **Recomputation**：直接把已生成 token 拼回 prompt 重算一遍 KV（一个 prompt phase 迭代搞定）。

论文 §7.3 结论：recomputation 的延迟开销最高不超过 swapping 的 20%，block size 16–64 时两者端到端相当。

### 4.5 常见误区：PagedAttention ≠ Continuous Batching

- **Continuous batching（iteration-level scheduling）来自 Orca**（OSDI 2022）：每次迭代粒度调度，完成的请求立刻移出、新请求插入。
- vLLM 是两者的**叠加**：iteration-level 调度（继承 Orca 思路）+ PagedAttention（本文贡献）+ 抢占/恢复机制。

**吞吐数字（注意分清口径）**：

- 论文口径（对比 FasterTransformer 和自实现的 Orca 变体）：相同延迟下吞吐 **2–4×**；ShareGPT trace 上可承受请求率为 Orca(Oracle) 的 1.7–2.7×、FasterTransformer 最高 22×；
- 官方博客口径：比 HuggingFace Transformers 最高 **24×**、比 HF TGI 最高 3.5×（论文本身没有 HF baseline，引用时注意）。

---

## 5. 稀疏注意力：让每个 query 少看一点

前两条线（FA、Paged）都不改变"每个 query 看所有 key"这个事实。稀疏注意力直接动计算量本身：**每个 query 只算一小部分 key 的注意力**，把 O(N²) 压向近线性。代价：从精确算法变成了近似算法（效果是否损失是核心问题）。

### 5.1 分类框架（两条正交的轴）

这是 NSA 论文（arXiv:2502.11089 §2）提出的划分，非常好用：

| | **固定模式（与内容无关）** | **动态选择（与内容/query 有关）** |
|---|---|---|
| **推理时才引入稀疏**（模型仍是 full attention 预训练的） | 滑窗截断 | StreamingLLM、H2O、Quest |
| **训练时就稀疏（natively trainable）** | Sparse Transformer、Longformer、BigBird | **NSA、MoBA、DSA** |

为什么要区分"推理时才稀疏"和"原生可训练"？NSA 论文的批评很犀利：推理时硬加稀疏会掉点（有研究表明 top 20% 的注意力只能覆盖约 70% 的注意力分数总和），而且很多方法是 **phase-restricted** 的（H2O 只在 decode 稀疏、prefill 仍要算全量），还可能与 GQA/MQA 不兼容（下面细说）。原生可训练方案让模型从一开始就学会"该看哪"。

### 5.2 静态稀疏（2019–2020 的经典工作）

- **Sparse Transformer**（OpenAI 2019，arXiv:1904.10509）：strided（隔行跨步）+ fixed（固定位置）两种分解模式，复杂度 O(n√n)；
- **Longformer**（AI2 2020，arXiv:2004.05150）：局部滑窗 + 任务驱动的全局 token（如 [CLS] 对全序列可见），随序列长度线性扩展；
- **BigBird**（Google 2020，arXiv:2007.14062）：随机 + 滑窗 + 全局三者组合，线性复杂度；理论贡献是证明了这种组合仍是**通用逼近器且图灵完备**——稀疏化没有损失理论表达能力。

### 5.3 推理侧动态稀疏 / KV 驱逐

- **StreamingLLM**（MIT 2023，arXiv:2309.17453，ICLR 2024）：发现 **attention sink** 现象——模型会把很高的注意力分数投给开头几个 token（即使语义不重要），作为"数值锚点"。纯滑窗 KV cache 一旦滚掉开头就 PPL 爆炸；保留**开头 4 个 token + 最近滑窗**即可泛化到 400 万 token，免训练。相比滑窗重计算基线最高 22.2× 加速。
- **H2O（Heavy Hitter Oracle）**（2023，arXiv:2306.14048，NeurIPS 2023）：发现一小撮 **heavy hitter** token 贡献了大部分注意力价值；用**累积注意力分数**动态保留 H2 + recent tokens 的 KV，驱逐其余。吞吐相比 FlexGen 等最高 29×。局限：只在 decode 稀疏。
- **Quest**（MIT 2024，arXiv:2406.10774，ICML 2024）：关键观察是 token 重要性**依赖 query**（不该像 H2O 那样驱逐后就再也回不来）。做法：KV 按 page 管理，每页维护 Key 的 **min-max 元数据**，query 来了先用元数据估计每页重要性上界，**只加载 top-K 页**做注意力——不驱逐，只选择性加载。self-attention 最高 2.23× 加速、延迟降 7.03×，精度损失可忽略。

### 5.4 原生可训练稀疏（2025 的主战场）

#### NSA（DeepSeek，arXiv:2502.11089，ACL 2025 Best Paper）

NSA（Native Sparse Attention）是 DeepSeek 面向下一代模型的稀疏架构，**三条并行分支 + learned gate**：

1. **Compression（粗粒度压缩）**：KV 按块经可学习 MLP 压缩成块级表示，捕捉全局轮廓；
2. **Selection（细粒度选择）**：重要性分数**复用压缩分支的注意力分数**（不用再算一遍 QK），top-n 选出关键 block 精确计算；
3. **Sliding Window（滑窗）**：专管局部上下文，防止模型"抄近路"全指望局部而学不好压缩/选择分支。

三个分支的输出由可学习门控（sigmoid）加权融合。

**硬件对齐设计**是 NSA 的另一半贡献（也是名字里 "hardware-aligned" 的含义）：块级稀疏对 Tensor Core 友好；专用 kernel 把 **GQA 组内所有 query head 一起载入 SRAM**，并强制**同组 head 共享同一份 block 选择**——这解决了一个实际问题：Quest 式逐 head 独立选择在 GQA 下会让组内各 head 选中 block 的并集很大，decode 时 KV 加载量降不下来。（GQA 背景见 [Attention 与 GQA](../../LLM/01-Architecture-%E6%A8%A1%E5%9E%8B%E6%9E%B6%E6%9E%84/Attention%20%E4%B8%8E%20GQA.md)）

**性能（论文口径，64k 序列）**：decode 提速 **11.6×**、前向 9.0×、反向 6.0×（对比 Full Attention），且 27B 规模预训练后通用基准/长上下文能力**持平或超过** Full Attention——这是"稀疏不掉点"的标志性结果。

#### MoBA（Moonshot AI / Kimi，arXiv:2502.13189，NeurIPS 2025）

MoBA 把 **MoE 的路由思想搬到注意力**：上下文切成 block，每个 block 的 K 做 mean-pooling 得到一个"块摘要"，query 与所有块摘要算内积得分，**top-k 门控**选出要看的 block，只对选中 block 做精确 softmax 注意力。

设计哲学是 "less structure"：滑窗注意力、attention sink 都是 MoBA 的特例（分别对应固定选最近块、固定选开头+最近块的门控），MoBA 不预设模式，让模型自己学看哪里；而且**同一份参数可以在 full/sparse 间无缝切换**。已部署于 Kimi 长上下文服务。

**性能（论文口径）**：1M token 时注意力层前向相比 Full Attention(FlashAttention) 最高 **6.5×**；扩展到 10M token（稀疏度 95.31%）时注意力计算时间最高减少 **16×**（注意：都是注意力层口径，非端到端）。

#### DSA（DeepSeek-V3.2，2025-09）

DSA（DeepSeek Sparse Attention）是 NSA 思想的产品化落地（随 DeepSeek-V3.2-Exp 发布，官方技术报告见 GitHub deepseek-ai/DeepSeek-V3.2-Exp）：

- **Lightning Indexer**：极轻量打分器（head 少、维度小、**FP8 计算**），为每个 query 给所有历史 token 打分；
- **细粒度 token 选择**：top-k（k=2048）选中后直接做 MLA 注意力，核心复杂度 O(L²) → **O(L·k)**。

128K 上下文下性能与 V3.1-Terminus 基本持平，推理成本显著下降（官方 API 价格随之腰斩）。社区共识：NSA 是三分支+门控的研究原型，DSA 是 token 级 top-k 选择的工程简化版。

### 5.5 稀疏注意力对比表

| 方法 | 年份 | 稀疏何时引入 | 选择粒度 | 是否丢 KV | 代表数字（论文口径） |
|---|---|---|---|---|---|
| Sparse Transformer | 2019 | 训练 | 固定模式 | — | O(n√n) |
| Longformer | 2020 | 训练 | 滑窗+全局 | — | 线性复杂度 |
| BigBird | 2020 | 训练 | 随机+滑窗+全局 | — | 线性；通用逼近器 |
| StreamingLLM | 2023 | 推理 | sink 4 + 滑窗 | 是 | 最高 22.2×；400 万 token |
| H2O | 2023 | 推理(decode) | token（累积分数） | 是 | 吞吐最高 29× |
| Quest | 2024 | 推理 | page（min-max 估计） | 否（选择性加载） | attention 2.23×，延迟 7.03× |
| **NSA** | 2025 | **训练** | 压缩+block top-n+滑窗 | 否 | 64k decode 11.6×；不掉点 |
| **MoBA** | 2025 | **训练** | block top-k（MoE 路由） | 否 | 1M 时 6.5×；已上 Kimi 生产 |
| **DSA** | 2025 | **训练**（continued） | token top-k（lightning indexer） | 否 | O(L·k)；V3.2 性能持平 |

---

## 6. 顺带一章：Ring Attention / Ulysses（多卡扩展序列）

前面所有技术都在单机范围内。**当序列长到单卡显存放不下 KV cache**时，需要序列维度的分布式并行（训练侧的模型并行背景见 [Megatron-LM 论文精读](../03-Training-%E8%AE%AD%E7%BB%83%E5%9F%BA%E7%A1%80%E8%AE%BE%E6%96%BD/Megatron-LM%20%E8%AE%BA%E6%96%87%E7%B2%BE%E8%AF%BB.md)）：

- **Ring Attention**（UC Berkeley，arXiv:2310.01889）：序列按 block 切到多卡，每卡持有本地 Q block；卡与卡组成环，**KV block 沿环流转**（发给下一台、收上一台），用 blockwise attention（与 online softmax 同源的块式累加，结果与块计算顺序无关）逐步累积输出。只要块计算时间 ≥ 块传输时间，通信完全被计算重叠，**不引入额外开销、不做近似**，上下文随设备数线性扩展（论文实验口径超过 1 亿 token）。
- **DeepSpeed Ulysses**（arXiv:2309.14509）：同一个问题的另一种切法——attention 前对 QKV 做一次 **all-to-all**，让每卡拿到**完整序列但只有一部分 attention head**（切 head 维而非序列维），算完再一次 all-to-all 换回来。通信量 O(N/P)，比 Megatron 序列并行小 P 倍；但**并行度不能超过 head 数**，GQA/MQA（KV head 少）场景受限。
- **与 FlashAttention 的关系**：正交且组合使用——Ring/Ulysses 管"序列切到哪台卡"，每台卡上的本地 attention 仍调用 FA kernel（社区标准实现 ring-flash-attention 即是如此）。
- **后续融合**：Striped Attention（arXiv:2311.09431，条纹切分解决 causal mask 下 Ring 的负载不均，最高 1.45×）；USP/YunChang（arXiv:2405.07719，把 Ulysses 和 Ring 组成 2D 网格混合使用，已被 NVIDIA TransformerEngine 采用）。

---

## 7. 全景总结：四条线如何叠加

回到开头的四问，用一张"正交关系图"收尾：

```
一次 decode 迭代中，一个 token 的注意力要经过四层优化：

[稀疏注意力]   这个 query 只需看 top-k 个 KV block（NSA/MoBA/DSA）
     ↓
[PagedAttention] 这些 block 在显存里物理上离散文放着，查 block table 找到（vLLM）
     ↓
[FlashAttention] 对每个选中的 block，用 IO-aware kernel 在 SRAM 里算精确注意力（FA2/3/4）
     ↓
[Ring/Ulysses]   如果序列跨多卡，KV 沿环流动或按 head all-to-all（训练/超长上下文）
```

几个值得记住的判断：

1. **FlashAttention 是精确算法**，稀疏注意力是近似算法——这是本质区别；
2. FA 的演进史就是一部"跟着 GPU 架构走"的历史：A100（IO 瓶颈）→ H100（异步/FP8）→ B200（非对称扩展，exp 都要软件模拟）。**AI Infra 的核心方法论：算法-硬件协同设计**；
3. PagedAttention 证明了**系统层的思想（OS 分页）可以直接改写 ML 系统的格局**——Infra 岗不只是写 kernel；
4. 稀疏注意力的趋势是从"推理时凑合"走向"原生可训练"（NSA/MoBA/DSA 在 2025 年集中爆发，且都已上生产），这大概率是下一代基座模型的标配；
5. 这些技术全部叠加，才撑起了今天百万级上下文、高并发的 LLM 服务。

## 8. 入行学习路径建议

1. **先读**：UW 手稿 *From Online Softmax to FlashAttention* → FA1/FA2 论文（顺序读，DefTruth 提醒 FA2 论文公式有少量笔误）→ vLLM 论文；
2. **再读**：FA3/FA4 论文 + Tri Dao 官方博客 → NSA/MoBA 对照读（两家同月发布，思路对照很有意思）；
3. **动手**：LeetCUDA（GitHub xlite-dev/LeetCUDA，3k+ star，含 FlashAttention 的 CUDA 示例实现）从写 kernel 开始；然后在 vLLM 源码里找 paged attention kernel；
4. **跟上**：Dao-AILab/flash-attention、vllm-project/vllm、MoonshotAI/MoBA、deepseek-ai 的 GitHub 仓库。

---

## 参考文献与信源

**论文**
- FlashAttention: https://arxiv.org/abs/2205.14135
- FlashAttention-2: https://arxiv.org/abs/2307.08691
- FlashAttention-3: https://arxiv.org/abs/2407.08608
- FlashAttention-4: https://arxiv.org/abs/2603.05451
- FlashDecoding++: https://arxiv.org/abs/2311.01282
- PagedAttention / vLLM (SOSP'23): https://arxiv.org/abs/2309.06180
- Sparse Transformer: https://arxiv.org/abs/1904.10509
- Longformer: https://arxiv.org/abs/2004.05150
- BigBird: https://arxiv.org/abs/2007.14062
- StreamingLLM: https://arxiv.org/abs/2309.17453
- H2O: https://arxiv.org/abs/2306.14048
- Quest: https://arxiv.org/abs/2406.10774
- NSA (ACL 2025 Best Paper): https://arxiv.org/abs/2502.11089
- MoBA: https://arxiv.org/abs/2502.13189 （代码 https://github.com/MoonshotAI/MoBA）
- DeepSeek-V3.2 / DSA: https://github.com/deepseek-ai/DeepSeek-V3.2-Exp
- Ring Attention: https://arxiv.org/abs/2310.01889
- DeepSpeed Ulysses: https://arxiv.org/abs/2309.14509
- Striped Attention: https://arxiv.org/abs/2311.09431
- USP/YunChang: https://arxiv.org/abs/2405.07719

**官方博客**
- FlashAttention-3 (Tri Dao): https://tridao.me/blog/2024/flash3/
- FlashAttention-2 (Hazy Research): https://hazyresearch.stanford.edu/blog/2023-07-17-flash2
- FlashDecoding (PyTorch): https://pytorch.org/blog/flash-decoding/
- vLLM 发布博客: https://blog.vllm.ai/2023/06/20/vllm.html
- FlashAttention 代码库: https://github.com/dao-ailab/flash-attention

**优质中文解读**
- DefTruth《从 Online-Softmax 到 FlashAttention V1/V2/V3》（知乎）: https://zhuanlan.zhihu.com/p/668888063
- 《显存管理革命：分页注意力机制 PagedAttention》（知乎）: https://zhuanlan.zhihu.com/p/1962435371720745525
- Big-Yellow-J《Kimi/DeepSeek 最新论文 MoBA 与 NSA 阅读》: https://www.big-yellow-j.top/posts/2025/02/21/Kimi-DS-Paper.html

---

*本文数字以原始论文口径为准；"60–80%"、"<4%"、"24× vs HF" 等标注为官方博客口径；个别社区转述数字已在文中注明。如发现错漏，以 arXiv 原文为准。*
