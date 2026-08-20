
> 来源：腾讯技术工程公众号《腾讯混元AI Infra如何优化Hy3 Preview：一次大模型推理性能提升的技术拆解》（作者：混元 AI Infra 推理团队）。本文是其结构化学习笔记，按「问题 → 方案 → 收益」组织。

## 目录

- [1. 背景与优化成果](#1-背景与优化成果)
- [2. 算子优化](#2-算子优化)
  - [2.1 Attention：动态调度负载均衡](#21-attention动态调度负载均衡)
  - [2.2 Router GEMM：双 BF16 重构 FP32](#22-router-gemm双-bf16-重构-fp32)
  - [2.3 FusedMoE：全算子流水线重构](#23-fusedmoe全算子流水线重构)
- [3. 算子融合](#3-算子融合)
  - [3.1 Fused RoPE+Norm+[Hadamard]+Quant+Store KV](#31-fused-ropenormhadamardquantstore-kv)
  - [3.2 Fused AllReduce+Norm+Add](#32-fused-allreducenormadd)
  - [3.3 采样融合算子](#33-采样融合算子)
  - [3.4 GEMM+Comm 细粒度通算融合](#34-gemmcomm-细粒度通算融合)
- [4. 并行策略](#4-并行策略)
  - [4.1 Prefill：TPSP](#41-prefilltpsp)
  - [4.2 Decode：DP+EP](#42-decodedpep)
- [5. 多级缓存](#5-多级缓存)
- [6. MTP 与异步调度优化](#6-mtp-与异步调度优化)
- [7. 量化与稀疏](#7-量化与稀疏)
  - [7.1 量化：GPTQ + Smooth + 旋转 + QAT](#71-量化gptq--smooth--旋转--qat)
  - [7.2 Stem 稀疏注意力](#72-stem-稀疏注意力)
- [8. 后续工作](#8-后续工作)
- [9. 相关页面](#9-相关页面)

---

## 1. 背景与优化成果

- **模型**：混元 Hy3 preview，腾讯新一代旗舰大模型，GQA + MoE 混合架构，原生支持 256K 超长上下文，面向 Agent、Coding 场景。
- **硬件约束**：主部署卡为 NVIDIA Hopper（96G），相比 Blackwell 算力低、显存紧凑、缺乏超节点互联，必须在有限硬件下把推理性能优化到极致。
- **SLO**：50ms TPOT，4s TTFT；精度 W8A8C8。
- **测试条件**：5000 条真实数据（最大输入 192k / 平均 68k；最大输出 64k / 平均 0.9k；缓存理论命中 80%）。
- **优化全景**：算子优化与融合、并行策略、多级缓存、MTP 与异步调度、量化与稀疏五大维度，多数算子已开源至 HPC-Ops 仓库。

![优化成果](../_assets/Inference-%E6%8E%A8%E7%90%86%E5%9F%BA%E7%A1%80%E8%AE%BE%E6%96%BD/hy3-01-benchmark.png)

---

## 2. 算子优化

### 2.1 Attention：动态调度负载均衡

**问题**：线上请求长度实时波动、batch 内长短混杂。传统静态 split-kv 需在长序列吞吐和短序列开销间做固定权衡——长序列需要更大的 split-kv 才能充分并行，短序列只需少量拆分，固定策略两头难兼顾。

**方案**：

- 所有请求按统一 Tile 粒度拆分，依据全局 Tile 总量均衡分配各 CTA 任务规模；
- 贪心装桶算法实现 Tile 任务极致均分，从源头杜绝计算单元负载失衡；
- Task Assign 模块在每次推理前生成专属任务映射表，各层 Attention Kernel 按映射表领取任务；
- Combine Kernel 统一合并 split-kv 结果。

![Attention 动态调度](../_assets/Inference-%E6%8E%A8%E7%90%86%E5%9F%BA%E7%A1%80%E8%AE%BE%E6%96%BD/hy3-02-attention-scheduler.png)

**收益**：单 batch 长文本场景单算子最高加速 **2.95x**；混合长度 batch 加速 **1.59x~1.76x**。

### 2.2 Router GEMM：双 BF16 重构 FP32

**问题**：MoE 路由及稀疏 Attention 等数值敏感模块中，BF16 激活 × FP32 权重面临效率与精度两难：权重降级损精度；激活升至 FP32/TF32 需逐元素类型转换，且 CUDA Core 算力带宽低、硬件利用率差。

**方案**：离线将 FP32 权重拆为高位 BF16 与低位残差 BF16 两组张量：

$$W \approx W_{\mathrm{high}} + \mathrm{scale} \times W_{\mathrm{low}}$$

（scale = 1/256，对齐 BF16 的 8 位尾数）。推理时执行两次 BF16 GEMM 并线性组合，激活全程 BF16、无需转换，均跑在 BF16 Tensor Core 上。实现上双路计算融合至单一 Kernel：输入只搬一次，双寄存器累加器分别缓存两路中间结果，Epilogue 阶段一次 FFMA 修正出高精度结果，全程无 HBM 往返。

![Router GEMM 双 BF16](../_assets/Inference-%E6%8E%A8%E7%90%86%E5%9F%BA%E7%A1%80%E8%AE%BE%E6%96%BD/hy3-03-router-gemm.png)

**收益**：N=192、K=4096 规格，M=2~4096 范围内相比 FP32（cuBLAS）加速 **2.86x~3.22x**。

### 2.3 FusedMoE：全算子流水线重构

将 MoE 完整推理链路五大核心阶段整合为一体化执行链路：

1. **路由与索引预处理**：共享内存分块统计，为每个专家预留连续显存输出区间，降低大规模 Token 下索引构建成本；
2. **Gate-Up GEMM**：路由索引直读原始输入，省去显式 Gather；取消 Warp Specialization，同一 Warp Group 完成搬运与计算，访存掩盖从 CTA 内软件流水升级为跨 CTA 硬件调度，提升 SM 驻留密度；
3. **激活量化 + Down GEMM**：量化结果按专家维度紧凑写入，保证 Down GEMM 顺序访存；
4. **Top-K 加权聚合**：推理末端加权求和，全程无额外 HBM 往返；
5. **PDL 无气泡串联**：全流程 PDL 构建无气泡执行链路，消除频繁 Kernel 启动开销。

**收益**：TP=8/EP=1 场景相比 vLLM CUTLASS、vLLM Triton、SGLang 加速 **1.5x~1.6x**；TP=1/EP=8 场景加速 **1.2x~1.5x**。

---

## 3. 算子融合

### 3.1 Fused RoPE+Norm+[Hadamard]+Quant+Store KV

**问题**：QKV Projection 之后存在连续 Element-wise 算子链（RoPE、RMSNorm、Hadamard 积、量化、KV Cache 写入）。各算子计算量小、算力强度低，频繁启动 Kernel 并反复读写 HBM 导致严重访存带宽受限，是 Prefill 阶段不可忽视的延迟来源。

**方案**：5 个算子深度融合为单一微型流水线 Kernel：

- **寄存器级数据流转**：中间结果全在寄存器暂存，消除中间变量（Norm 后、RoPE 后）的 HBM 存取；
- **在线量化与 KV Cache 存储**：写入前完成在线量化，直接以低比特格式写显存，压缩写出带宽。

**收益**：融合算子加速约 **5x**。

### 3.2 Fused AllReduce+Norm+Add

**问题**：张量并行下通信、残差计算、归一化拆分执行造成性能损耗。

**方案**（联合腾讯网络平台部）：将三者全链路融合为 NVLink 原生一体化操作 `RMSNorm(AllReduce(x) + residual, weight)`，基于 CUDA 多播与 P2P，支持 BF16 及单机多卡，采用 Two-shot 策略：

- 高吞吐版本（`fuse_allreduce_rmsnorm_high_throughput`）：依托 NVSwitch 多播完成归约，适配大 Token 量 Prefill；
- 低延迟版本（`fuse_allreduce_rmsnorm_low_latency`）：基于 Lamport P2P，通过 PDL 双 Kernel 重叠执行，适配小批量 Decode。

**收益**：覆盖 8~32k tokens 场景，相比 NCCL 与 FlashInfer 同类路径最高加速 **1.68x**。

### 3.3 采样融合算子

**问题**：传统采样后处理链由十余个零碎 Kernel 串联（重复惩罚、温度缩放、Top-K、Top-P、Softmax、随机采样等），流程碎片化。每个 Kernel 独立访问全局词表（vocab_size 级别），HBM 加载次数线性膨胀；重复惩罚的掩码数据还需 CPU-GPU 拷贝，引入额外同步。

**方案**：10 余个零碎 Kernel 融合为 2 个核心 CUDA Kernel，封装为单一 `fused_sampler` 算子；针对差异化场景（简单温度采样 / 完整采样）算子内自动适配专用内核：

- **全词表单次加载**：全局词表 GPU 读取压缩至 1 次，计算与访存充分掩盖；
- **GPU 闭环惩罚计算**：重复惩罚掩码在 GPU 内完成，消除 CPU-GPU 拷贝；
- **细粒度多 CTA 并行**：单请求拆到多线程块并行，提升小 Batch 下 SM 并发度；
- **局部堆归并 Top-K**：Max Top-K ≤ 64 时用局部堆归并替代全局阈值扫描/拒绝采样，避免全词表重复读；
- **Top-K 与 Softmax 融合**：Top-K 归约与 Softmax 的 max/sum 合并。

| 融合前 | 融合后 |
| --- | --- |
| ![融合前](../_assets/Inference-%E6%8E%A8%E7%90%86%E5%9F%BA%E7%A1%80%E8%AE%BE%E6%96%BD/hy3-04-sampler-before.png) | ![融合后](../_assets/Inference-%E6%8E%A8%E7%90%86%E5%9F%BA%E7%A1%80%E8%AE%BE%E6%96%BD/hy3-05-sampler-after.png) |

**收益**：相比 vLLM 与 FlashInfer 的采样算子提升约 **5.5x / 2.5x**。

### 3.4 GEMM+Comm 细粒度通算融合

**问题**：Prefill TPSP 并行场景下 GEMM 与 ReduceScatter 串行执行，通信暴露。

**方案**：SM 资源显式划分为计算 SM（矩阵乘）与通信 SM（RS 搬运）两类角色。计算 SM 每产出一个 Tile 即落盘本地 Buffer，信号量通知通信 SM 立即对就绪分片发起 RS，实现 Tile 级计算通信重叠。在传统 Load Warp 与 MMA Warp 之外特化专职 **Epilogue Warp**，形成 `Load → MMA → Epilogue` 三级流水：

1. **Load Warp**：异步预取下一轮 Tile；
2. **MMA Warp**：累加完仅写回 SMEM，立即进入下一轮计算；
3. **Epilogue Warp**：异步取 SMEM 结果，完成 Quant/Scale 后处理写回 HBM，最后触发通信 SM 就绪信号。

![GEMM+Comm 通算融合](../_assets/Inference-%E6%8E%A8%E7%90%86%E5%9F%BA%E7%A1%80%E8%AE%BE%E6%96%BD/hy3-06-gemm-comm-fusion.png)

**收益**（与腾讯网络平台部联合优化）：

| 矩阵形状 (M, N, K) | GEMM | RS | 串行 | 本方案 | 覆盖率 | vs 串行 |
| --- | --- | --- | --- | --- | --- | --- |
| (8k, 4096, 1024) | 257.74 | 250.93 | 560.12 | 316.63 | 76.5% | 1.77× |
| (16k, 4096, 1024) | 512.04 | 480.82 | 1,096.59 | 604.98 | 80.7% | 1.81× |
| (32k, 4096, 1024) | 1,016.46 | 945.63 | 1,962.30 | 1,162.79 | 84.5% | 1.69× |
| (64k, 4096, 1024) | 2,026.13 | 1,846.68 | 3,872.58 | 2,307.68 | 84.8% | 1.68× |

---

## 4. 并行策略

### 4.1 Prefill：TPSP

**问题**：纯 TP8 方案三重代价：

1. **冗余计算**：Elementwise / Router 等 token-wise 算子沿完整序列在各卡重复执行；
2. **通信过重**：AllReduce 在 8 卡间交换全量数据；
3. **算子畸形**：MoE Grouped GEMM 沿 hidden 维切至极窄 shape，计算效率急剧退化。

**方案**：保持单机 8 卡部署与模型精度不变，**SP 拆分 + 通算融合 + 通信量化 + 并行模式调整**四项技术组合，系统性压缩 TTFT。

![TPSP 方案](../_assets/Inference-%E6%8E%A8%E7%90%86%E5%9F%BA%E7%A1%80%E8%AE%BE%E6%96%BD/hy3-07-tpsp.png)

**收益**：

| 场景 | 优化前 TTFT | 优化后 TTFT | 节省 | 降幅 |
| --- | --- | --- | --- | --- |
| Prefill 16k | 764 ms | 536 ms | −228 ms | −29.9% |
| Prefill 32k | 1885 ms | 1424 ms | −461 ms | −24.5% |

### 4.2 Decode：DP+EP

**问题**：单机部署面临存算双重瓶颈：

1. 显存被权重占用近半，挤压 KV Cache 空间，制约最大并发数；
2. 小 Batch 下 MoE Grouped GEMM 算力强度低，严重 Memory-bound。

**方案**：Attention DP + MoE EP 跨节点混合并行。增大 EP Size 实现权重多机分布式存储，腾出显存转产 KV Cache 吞吐；跨节点聚合 Batch Size 使 Grouped GEMM 进入 Compute-bound 区，最大化 Tensor Core 利用率。

- 自研 HPC Kernel（gate、route、group GEMM、count and gather、combine），Hopper 上 SOTA；
- 长序列 Attn DP+TP 混合策略，大幅降低 DP 负载不均衡影响，只承担少量机内通信开销；
- **异步专家负载均衡（Async EPLB）**：NCCL P2P 异步执行权重重排，每步仅一层权重的通信隐藏于前向计算之后，权重重排与 Decode 完全重叠；
- shared expert 拆分与 dispatch、combine 并行，通信与计算 overlap。

![DP+EP 架构](../_assets/Inference-%E6%8E%A8%E7%90%86%E5%9F%BA%E7%A1%80%E8%AE%BE%E6%96%BD/hy3-08-dp-ep.png)

**收益**：端到端吞吐提升 **15.7%~44.7%**。

---

## 5. 多级缓存

**问题**：Agent、Coding 场景大量长上下文、多轮对话和可复用公共前缀，Prefill 开销直接影响 TTFT 与吞吐。但 Prefix Cache 仅靠 GPU 显存面临四重瓶颈：

1. **容量受限**：权重与运行时占用后，可缓存显存极有限；
2. **淘汰加速**：长文请求的大体量 KV Cache 加速已有缓存淘汰；
3. **重复计算**：缓存淘汰后相同前缀需重新 Prefill；
4. **跨实例不可复用**：单机缓存在实例迁移、扩缩容、跨节点调度时完全失效。

**方案**：构建 **GPU → CPU → KVStore** 三级缓存体系，把 KV Cache 从单一显存短期缓存扩展为分层存储、按需加载、跨请求复用的多级架构，不增加 GPU 显存占用即显著扩大有效缓存容量。

**调度流程**：请求按 L1→L2→L3 顺序查询可复用前缀，命中后按需加载回 GPU 并跳过对应 Prefill；新生成的完整 Block 按策略异步下沉至 L2/L3 供后续复用。

![多级缓存体系](../_assets/Inference-%E6%8E%A8%E7%90%86%E5%9F%BA%E7%A1%80%E8%AE%BE%E6%96%BD/hy3-09-multi-level-cache.png)

---

## 6. MTP 与异步调度优化

**问题**：传统异步调度基于「每轮稳定生成 1 个 token」的假设，GPU 计算时 CPU 提前准备下一轮输入以掩盖 CPU 耗时。多层 MTP 引入动态接收长度——下一轮序列长度、位置编码、KV Cache 映射均强依赖验证结果。传统做法需在 GPU Forward 验证结束后强制同步、拷回 CPU 再准备下一轮，CPU 准备只能与 MTP 层 Forward 重叠；而 MTP 层计算极快，无法充分掩盖 CPU 耗时。

**方案**：解除 CPU 对真实接收长度的同步依赖——数据准备阶段一律按最大接收长度更新状态并组装下一轮输入；下一轮实际计算前，再以上一轮真实验证结果修正计算所依赖的关键值。CPU 可提前一轮完成准备与 Launch，无需阻塞等 GPU 结果。

![MTP 异步调度](../_assets/Inference-%E6%8E%A8%E7%90%86%E5%9F%BA%E7%A1%80%E8%AE%BE%E6%96%BD/hy3-10-mtp-async.png)

**收益**：减少 decode 间 5~10ms CPU 气泡，端到端提升 **10%~20%**。

> [!NOTE]
> 同一思想在普通解码循环（无投机采样）中的应用与完整工程实现，见 [消除 GPU 气泡：流水线化解码（Photon）](%E6%B5%81%E6%B0%B4%E7%BA%BF%E5%8C%96%E8%A7%A3%E7%A0%81%EF%BC%9A%E6%B6%88%E9%99%A4%20GPU%20%E6%B0%94%E6%B3%A1.md)。

---

## 7. 量化与稀疏

### 7.1 量化：GPTQ + Smooth + 旋转 + QAT

**问题**：直接用 W4A8 + Attn FP8 量化虽大幅压缩体积，但权重极低比特表示与激活离群值会严重放大量化误差，精度显著退化。

**方案**（基于腾讯开源框架 [AngelSlim](https://github.com/Tencent/AngelSlim)）：「GPTQ 权重重建 + 激活平滑与旋转变换 + QAT 轻量化微调」三级联合优化，系统性消除 Attn FP8 + W4A8 下的精度损失：

1. **GPTQ 逐层权重重建**：基于 Hessian 逆的逐层误差补偿，降低 INT4 权重量化的精度损失；
2. **激活平滑（Smooth）**：搜索逐通道平滑因子，不改变输出前提下压缩激活离群值幅度，收窄动态范围；
3. **Hadamard 在线旋转变换（Attention 专用）**：Q/K 量化前施加正交旋转，把集中于少数通道的离群值均匀打散；计算高效、可融合进推理 Kernel，几乎零额外开销（与 3.1 节融合算子链中的 Hadamard 对应）；
4. **QAT 轻量化微调**：训练中模拟 W4A8 量化噪声，仅更新量化相关参数（scale/zero-point），收敛快、训练成本极低。

![量化方案](../_assets/Inference-%E6%8E%A8%E7%90%86%E5%9F%BA%E7%A1%80%E8%AE%BE%E6%96%BD/hy3-11-quantization.png)

**收益**：多领域评测集与 BF16 基线持平或差距 < 1%（精度无损）；端到端吞吐提升 **28%+**。

### 7.2 Stem 稀疏注意力

**问题**：256K 上下文下标准自注意力的二次方复杂度使 Prefill 延迟和显存开销随序列长度急剧增长，成为 TTFT 关键瓶颈。

**方案**：自研 Stem 稀疏注意力算法 + HPC-BSA（Block Sparse Attention）算子，仅用 **25%** 计算预算达到接近稠密注意力的精度。核心思路：从因果注意力的信息流视角重新审视「哪些 token 该保留、哪些该剪枝」。

![Stem 整体流程](../_assets/Inference-%E6%8E%A8%E7%90%86%E5%9F%BA%E7%A1%80%E8%AE%BE%E6%96%BD/hy3-12-stem-flow.png)

两大关键技术：

1. **Token Position-Decay（位置衰减策略）**
   - 洞察：因果架构中序列头部 token 参与所有后续聚合计算，误差逐层递归放大；尾部 token 影响仅限局部；
   - 做法：每个 query 位置的 Top-k 预算从头部 `k_start` 线性衰减到尾部 `k_end = μ · k_start`。头部关键 token 获更大预算保护递归依赖链，尾部冗余 token 激进剪枝；
   - 效果：总预算不变，仅重新分配即显著提升精度。

2. **Output-Aware Metric（输出感知度量，OAM）**
   - 洞察：传统方法仅按注意力分数（QK^T）选 token，但注意力分数只反映「路由概率」而非实际「信息贡献」——高注意力但 Value 模长接近零的 token 对输出几乎无贡献；
   - 做法：将 Value 向量模长作为信号强度引入选择标准：

$$M(i,j) = QK^{\top} + \beta \cdot \max\left(0,\ \log \lVert V_{j} \rVert_{2}\right)$$

   - 优势：对数变换使其可直接复用标准 Top-k 内核，无额外计算开销。

方案已集成至 [AngelSlim](https://github.com/Tencent/AngelSlim)。

**收益**：LongBench v2、CL-bench、SWA 等数据集上取得与密集注意力相当的精度；128K 上下文首字耗时（TTFT）提升 **3.6 倍**。

![Stem 精度对比](../_assets/Inference-%E6%8E%A8%E7%90%86%E5%9F%BA%E7%A1%80%E8%AE%BE%E6%96%BD/hy3-13-stem-result-1.png)
![Stem 性能收益](../_assets/Inference-%E6%8E%A8%E7%90%86%E5%9F%BA%E7%A1%80%E8%AE%BE%E6%96%BD/hy3-14-stem-result-2.png)

---

## 8. 后续工作

- 长上下文下仍有显存瓶颈，推进 **C4 与 W4** 相关优化，精度无损前提下进一步提升吞吐；
- 探索全新**并行投机解码**：保证接收率的同时以更低计算代价产出更多投机 Token；
- 持续优化：调度与并行策略、PD 高效传输、多级缓存中心、跨机通信与流量控制；
- 适配其他硬件平台，进一步降低推理成本。

---

## 9. 相关页面

- [KV Cache 内存管理（PagedAttention）](KV%20Cache%20%E5%86%85%E5%AD%98%E7%AE%A1%E7%90%86%EF%BC%88PagedAttention%EF%BC%89.md) — 多级缓存体系建立在 PagedAttention 的 Block 化管理之上
- [vLLM 论文精读](vLLM%20%E8%AE%BA%E6%96%87%E7%B2%BE%E8%AF%BB.md)、[推理全流程串讲（概览篇）](LLM%20%E6%8E%A8%E7%90%86%E5%85%A8%E6%B5%81%E7%A8%8B%E6%A6%82%E8%A7%88.md) — Prefill/Decode 分离、TTFT/TPOT 等指标背景
- [Attention 与 GQA](../../llm/01-Architecture-%E6%A8%A1%E5%9E%8B%E6%9E%B6%E6%9E%84/Attention%20%E4%B8%8E%20GQA.md) — GQA 架构与 split-kv 的背景
- [MoE学习笔记](../../llm/01-Architecture-%E6%A8%A1%E5%9E%8B%E6%9E%B6%E6%9E%84/%E6%B7%B7%E5%90%88%E4%B8%93%E5%AE%B6%E6%A8%A1%E5%9E%8B%EF%BC%88MoE%EF%BC%89.md) — MoE 路由、专家并行的模型侧基础
- [通信原语与 NCCL](../02-Foundations-%E5%9F%BA%E7%A1%80%E6%A6%82%E5%BF%B5/%E9%80%9A%E4%BF%A1%E5%8E%9F%E8%AF%AD%E4%B8%8E%20NCCL.md) — AllReduce / ReduceScatter / P2P 等通信原语
- [LLM 推理加速与算子优化学习路线](../01-Roadmaps-%E5%AD%A6%E4%B9%A0%E8%B7%AF%E7%BA%BF/LLM%20%E6%8E%A8%E7%90%86%E5%8A%A0%E9%80%9F%E4%B8%8E%E7%AE%97%E5%AD%90%E4%BC%98%E5%8C%96%E5%AD%A6%E4%B9%A0%E8%B7%AF%E7%BA%BF.md) — 本文是该路线各主题的工业界综合实践案例
