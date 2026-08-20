
## 目录

- [1. 总览：三条主线](#1-总览三条主线)
- [2. GPU/Training Systems](#2-gputraining-systems)
  - [2.1 DITRON：分布式多级 Tiling 编译器](#21-ditron分布式多级-tiling-编译器)
  - [2.2 Quartet II：NVFP4 精确预训练](#22-quartet-iinvfp4-精确预训练)
  - [2.3 Untied Ulysses / UPipe：按 Head 分块的上下文并行](#23-untied-ulysses--upipe按-head-分块的上下文并行)
- [3. LLM Serving](#3-llm-serving)
  - [3.1 PPD：区分 Full/Append Prefill 的 PD 分离](#31-ppd区分-fullappend-prefill-的-pd-分离)
  - [3.2 AMPD：多轮推理的自适应 PD 路由](#32-ampd多轮推理的自适应-pd-路由)
  - [3.3 Beyond Prediction / UniBoost：尾延迟感知调度](#33-beyond-prediction--uniboost尾延迟感知调度)
- [4. Agent Infra](#4-agent-infra)
  - [4.1 ThunderAgent：以 Agentic Program 为调度对象](#41-thunderagent以-agentic-program-为调度对象)
  - [4.2 CONCUR：基于拥塞控制的 Agent 并发调节](#42-concur基于拥塞控制的-agent-并发调节)
  - [4.3 GraphFlow：基于图的 Agent Workflow 管理](#43-graphflow基于图的-agent-workflow-管理)
- [5. 相关页面](#5-相关页面)

---

## 1. 总览：三条主线

ICML 2026 的 AI Infra 形成三条清晰主线：

- **GPU / Training System**：让模型结构、编译器、通信和 GPU 集群真正协同；
- **LLM Serving**：从单纯追求吞吐，转向围绕 KV Cache、长上下文和异构请求做全局资源调度；
- **Agent Infra**：开始讨论 Workflow、编译、状态、恢复和安全。

背后的信号：AI Infra 正在从「把 GPU 跑满」走向系统级的全局执行优化。

## 2. GPU/Training Systems

趋势判断：**通信正在成为编译器的对象**（DITRON 把计算、通信、同步联合编译）；**训练算法直接适配硬件数据格式**（Quartet II 围绕 Blackwell NVFP4 重新设计无偏梯度量化）；**并行策略与模型结构共同设计**（Head Parallel、UPipe、异步 Pipeline——MoE、长上下文、Dense Transformer 的最佳切法各不相同，不再有万能并行方案）。

![GPU/Training Systems 论文分布](../_assets/Foundations-%E5%9F%BA%E7%A1%80%E6%A6%82%E5%BF%B5/icml26-01-training-overview.jpg)

### 2.1 DITRON：分布式多级 Tiling 编译器

[DITRON: Distributed Multi-level Tiling Compiler for Parallel Tensor Programs](https://arxiv.org/abs/2605.02953)（ByteDance Seed、北大、清华、浙大、上交）

**背景**：大模型训练的计算由 Triton/cuBLAS 优化，跨 GPU 通信交给 NCCL——两边各自很快，但编译器看不到完整数据流，无法把计算和通信编排到一起。规则矩阵乘时代问题不大（算完整个算子再统一 AllReduce）；但 MoE、长上下文、多维并行时代，算子中间结果需要一边算一边经 NVLink/RDMA 发出，等整个算子完成再通信会让计算单元和网络轮流闲置。

**方案**：把分布式训练编译成一个统一的数据流程序——小块数据算完立刻发送，对端收到即可开始后续计算。

- **前端三层抽象**：Core Tile（GPU 内计算）、Device Chunk（跨设备数据搬运）、Task Tile（模型级任务依赖）——三者粒度和硬件机制完全不同，不能压进同一个「算子」里；
- **中端**：distributed swizzling（把大计算、大通信打碎，在时间线上交错编排）、依赖分析、wait/notify 插入，决定每个 tile 的计算顺序、通信时机和 rank 映射，形成通算流水线；
- **后端**：Distributed IR 映射到 NVSHMEM / rocSHMEM / NVLink / RDMA 等硬件原语，同一程序生成面向 NVIDIA 或 AMD 集群的执行代码。生成的 kernel 可直接远程读写其他 GPU 显存（MegaKernel / Persistent Kernel 思路），并用 DMA/TMA 信号量做硬件级控制流。

![DITRON 架构](../_assets/Foundations-%E5%9F%BA%E7%A1%80%E6%A6%82%E5%BF%B5/icml26-03-ditron-framework.jpg)

**意义**：把分布式通信从「外部库调用」变成编译器可分析、可变换、可代码生成的程序对象，为「分布式 Triton」提供框架。产业上可能启发新的训练方式：训练框架只刻画模型，分布式编译器消化跨 GPU 执行与高效通信的实现难题。

### 2.2 Quartet II：NVFP4 精确预训练

[Quartet II: Accurate LLM Pre-Training in NVFP4 by Improved Unbiased Gradient Estimation](https://arxiv.org/abs/2601.22813)（IST Austria、Red Hat AI）

**背景**：Blackwell 提供原生 NVFP4 Tensor Core，但把训练从 BF16 降到 4 bit，量化误差可能在更新中累积。现有两条路线各有硬伤：

- **Round-to-Nearest（RTN）**：单次误差小，但给梯度引入系统偏差；
- **Stochastic Rounding（SR）**：按距离随机舍入、理论上无偏，但 FP4 刻度太稀疏，每元素随机跳动带来很大方差。

两难：RTN 低误差但有偏；SR 无偏但噪声大。Quartet II 的洞察：**前向和反向在解决两个不同的问题，不必用同一种量化**——前向算 loss 需要精确（RTN）；反向算梯度是长期修正过程，容忍单次不精确但不能有系统性偏差（可引入随机性保无偏）。

**方案**：前向反向分开设计。

- **前向**：原生 1×16 NVFP4 scale（每 16 个 FP4 共享一个 scale，硬件友好且局部精度好）+ **Four-over-Six**（每块数据分别用量程 4 和量程 6 两套范围编码，选误差更小的一套——量程 4 精度细、量程 6 覆盖大）；主权重保持 FP32，仅在 GEMM 前量化；
- **反向**：核心创新 **MS-EDEN**——先用 **Hadamard 旋转**打散梯度异常值，再 RTN 量化到 FP4；为保无偏，不再随机动每个 FP4 数值，而是把 EDEN 修正量写进每 16 数共享的 FP8 scale、**只对 scale 做随机舍入**，方差大幅降低；
- 权重分支反向时沿相同内维做匹配旋转，矩阵乘中两侧的旋转自然抵消。

![Quartet II 计算流](../_assets/Foundations-%E5%9F%BA%E7%A1%80%E6%A6%82%E5%BF%B5/icml26-05-quartet2-scheme.jpg)

**意义**：较完整地回答了「FP4 原生训练怎样落地」——Blackwell NVFP4 有了实现思路；学界上，低精度训练中误差结构、梯度统计性质与硬件执行路径的协同设计可能成为新方向。

### 2.3 Untied Ulysses / UPipe：按 Head 分块的上下文并行

[Untied Ulysses: Memory-Efficient Context Parallelism via Headwise Chunking](https://arxiv.org/abs/2602.21196)（Together AI）

**背景**：长上下文训练中，序列长度 S 增长使显存沿多条路径上升：FlashAttention 已解决 Attention score 的平方级 HBM 驻留，但**下一道显存墙是全量 QKV 与 All-to-All 通信 buffer 同时驻留**（DeepSpeed-Ulysses 一次性为全部 H 个 head 生成 QKV，长序列下容易 OOM）。关键判断：长序列本身已提供足够大的计算量，没必要让全部 head 同时驻留显存。

**方案**：**UPipe** 把一次完成的 Attention 拆成多个阶段，每阶段只处理一小组 head：依次完成 QKV 投影 → 输入 All-to-All → FlashAttention → 输出 All-to-All；处理完后下一组 head **直接复用同一组 QKV 和通信 buffer**。DeepSpeed-Ulysses 的中间显存取决于全部 H 个 head，UPipe 只取决于当前组的 U 个 head；U 取 GPU 数量时显存开销甚至不再随总 head 数增长。还设计了兼容 [GQA](../../llm/01-Architecture-%E6%A8%A1%E5%9E%8B%E6%9E%B6%E6%9E%84/Attention%20%E4%B8%8E%20GQA.md) 的调度：先发送不同的 KV head，再调整 Query head 执行顺序，让后续阶段复用已通信的 K、V，避免重复传输。

![DeepSpeed-Ulysses vs UPipe](../_assets/Foundations-%E5%9F%BA%E7%A1%80%E6%A6%82%E5%BF%B5/icml26-07-upipe-vs.jpg)

两者数学计算相同，UPipe 把峰值显存从「全部 head 的中间张量」降到「当前一组 head」。

**意义**：不要求更多 GPU、不重度依赖 CPU offloading，把 Ulysses 的高吞吐与分块方法的低显存结合；可直接用于现有 TorchTitan / FlashAttention / Context Parallelism 软件栈，用相同硬件训练更长序列。Sequence Chunking × Head Chunking × Hybrid CP 的多维联合设计可能是新方向。

## 3. LLM Serving

脉络回顾：[Orca](https://www.usenix.org/conference/osdi22/presentation/yu) 迭代级调度 → [vLLM/PagedAttention](https://arxiv.org/abs/2309.06180) 解决 KV Cache 碎片 → [Sarathi-Serve](https://arxiv.org/abs/2403.02310) Chunked Prefill 缓解长 prompt 阻塞 decode → [DistServe](https://arxiv.org/abs/2401.09670) P/D 分离 → [Mooncake](https://arxiv.org/abs/2407.00079) 把 KV Cache 扩展成跨 HBM/DRAM/SSD/网络的数据层。

ICML 2026 的三个趋势：**Request-centric → Phase-aware**（Prefill/Decode 二分仍太粗，需细分 full prefill / append-prefill / decode / verification / tool-call resume / cache reload 等阶段）；**Stateless → Stateful**（Agent、LoRA adapter 与 KV Cache 不应独立处理——共享 backbone 和历史的多个 Agent，KV 应拆成共享基础状态 + adapter 差异状态）；**长度预测 → Risk-aware Scheduling**（即使准确预测 decode 长度，SRPT 在突发流量、KV 显存压力、高抢占成本下仍会有糟糕的 P99）。

![ICML 2026 Serving 论文分布](../_assets/Foundations-%E5%9F%BA%E7%A1%80%E6%A6%82%E5%BF%B5/icml26-08-serving-overview.jpg)

### 3.1 PPD：区分 Full/Append Prefill 的 PD 分离

[Not All Prefills Are Equal: PPD Disaggregation for Multi-turn LLM Serving](https://arxiv.org/pdf/2603.13358)（University of Chicago）

**背景**：DistServe/Splitwise/Mooncake 把 Prefill 与 Decode 拆到不同 GPU Pool，但**所有 Prefill 都先走 P Pool 再传 KV 给 D Pool**，不区分全量与增量 prefill。多轮对话中若 KV 已在 Decode 节点，仍强制走 P→D 就是重复计算 + 重复传输。核心问题变成「计算应该靠近哪份状态」。

**方案**：Full Prefill（完整上下文、计算重）与 Append-Prefill（只处理新增 token、复用历史 KV、干扰小）区别对待。多轮请求到达时，根据 workload、P:D 资源配置及 TTFT/TPOT 的 SLO 权重**动态选择**：D-local Append-Prefill（复用本地历史 KV，省去重算与 P→D 传输）或 P 远程执行（避免抢占繁忙的 Decode GPU）。方法上先离线 profiling 各负载与资源配置的 TTFT–TPOT 权衡，再由在线 Router 决策。

![PPD 动态路由](../_assets/Foundations-%E5%9F%BA%E7%A1%80%E6%A6%82%E5%BF%B5/icml26-10-ppd-routing.jpg)

**意义**：学术上把调度粒度从 request-aware、phase-aware 推进到 **phase-subtype-aware 与 state-aware**；工业上可作为 vLLM/SGLang Router 中的一个策略落地，无需改模型。

### 3.2 AMPD：多轮推理的自适应 PD 路由

[Efficient Multi-round LLM Inference over Disaggregated Serving](https://arxiv.org/pdf/2602.14516)（东南大学、剑桥、北大、蚂蚁、上交）

**背景**：多轮 Agent 的执行是 Initial Prefill → Decode → Tool → Incremental Prefill → Decode，Prefill 反复出现。本地增量 prefill 复用 D 节点历史 KV、降 TTFT，但会暂停 Decode（伤 ITL）；远程执行保护 ITL，却增加 P 队列与双向 KV 传输。固定 colocate 或固定 disaggregate 都不是答案——真正的问题是**每个增量 Prefill 到达时，P 和 D 哪侧还有更多 SLO 余量**。

**方案**（与 PPD 出发点类似，更工程化）：

1. **Request Binding**：请求绑定一个 Decode Worker 作为该 Session 的 KV owner，各轮次状态位置明确；
2. **Adaptive Routing**：按 P 侧 TTFT slack、D 侧 ITL slack、两边队列、Prefill 计算量与 KV 传输成本，选 P 远程或 D 本地（P 不忙走 P；P 忙但 D 有余量走 D；都忙则算哪条总体更便宜）；
3. **Prefill Reordering**：不严格 FIFO，优先执行最可能违反 TTFT SLO 的任务，避免长任务饿死；
4. **Offline Deployment Planner**：按模型、GPU、网络和多轮 workload 决定 P/D replica 数量与并行度，避免路由合理但底层资源比例失衡。

![AMPD 系统概览](../_assets/Foundations-%E5%9F%BA%E7%A1%80%E6%A6%82%E5%BF%B5/icml26-12-ampd-overview.jpg)

**与 PPD 对比**：PPD 重在学术结论（Full 与 Append Prefill 不是同一种负载）；AMPD 重在工程化（路由、重排、资源规划把整个系统管起来）。

**意义**：以用户体验指标（TTFT/ITL）驱动增量任务放置，而不单看 GPU 利用率；可作用于 Dynamo、vLLM 等 PD Serving 控制面。

### 3.3 Beyond Prediction / UniBoost：尾延迟感知调度

[Beyond Prediction: Tail-Aware Scheduling for LLM Inference](https://arxiv.org/pdf/2606.18431)（Cornell、Microsoft Azure Research、NVIDIA）

**背景**：在线 Serving 混入短对话、长推理、突发请求时 P95/P99 延迟失控。现有调度三类各有短板：FCFS 保护早到与长请求但易被超长任务堵住；SJF/SRPT 预测长度优先短任务但饿死长任务（99 个 10 秒任务插队，10 分钟任务等 20 分钟——均值漂亮、P99 崩溃）；LAS/MLFQ 不预测长度但新请求持续抢占会制造更多活跃 KV Cache。即使准确知道输出长度，SRPT 优化的也只是平均延迟，无法天然控制 P99；且 LLM 抢占会引发 KV 换出、重算与显存压力。

**方案**：**UniBoost**——请求到达时获得 soft priority boost（临时优先级加成，跑久后优势消失，不能一直压老任务），核心参数 γ 在 FCFS（保护长任务）与 LAS/SJF（快速完成短任务）之间调节。四个阶段：

1. **DistBoost**（对照设计）：Prefill/Decode 两个队列内各自 boost 排序；
2. **UniBoost-Base**：把 Prefill 和 Decode 映射到**同一 soft-priority 空间**，按「到达时间 − work-dependent boost」统一选下一批，打破「Decode 永远优先」；
3. **MemGuard**：从显存成本角度判断抢占是否划算——优先级更新限制在离散 work threshold（如每 16/32/64 token 一个门槛）+ minimum-run hysteresis（迟滞防抖），避免每生成一个 token 就抢占搬 KV；
4. **γ-Ada**：统计近期 P95/P99 TTFT/TTFT/TBT，按实际尾部分布自适应更新 γ（长请求卡顿就多保护，短请求被堵就提优先级）。

运行时真正工作的是 Phase 2 统一优先级 + Phase 3 MemGuard + Phase 4 γ-Ada。UniBoost 管「时间与延迟 SLO」，MemGuard 管「空间与显存成本」，两者闭环。

**意义**：学术上把 LLM 调度转化为「联合控制尾延迟、抢占和 KV 状态」的问题，把排队论 soft priority 与 continuous batching、P/D、KV 约束连接起来；工业上是本地 scheduler 与 KV eviction 模块的有效思路，尤其适合 reasoning 请求、突发流量、显接近饱和场景。

## 4. Agent Infra

Agent Infra 面对长期运行、持续积累状态、反复调用工具的 Agent Program。两个趋势：**固定策略 → 运行时闭环控制**（CONCUR 按 KV 压力调活跃 Agent 数；BudgetMem 按 query 难度分配 memory 预算；R³DAO 按故障信息改局部工作流）；**完整私有状态 → 共享公共状态 + 少量差异**（LRAgent 共享基础 KV Cache；GraphFlow 共享 Workflow Operation 和状态残差）。

七层划分：Workflow Representation（GraphFlow）、Compilation（Agent JIT、EvoC2F）、Execution Runtime（ThunderAgent、CONCUR）、State and Memory（LRAgent、BudgetMem）、Recovery（R³DAO）、Observability（AgentXRay）、Security（SandboxEscapeBench）。核心 insight：瓶颈正从模型计算转向 Workflow、模型状态、工具状态、记忆状态和失败状态之间的**协调**——效率评估要看多少工作被重复执行、多少状态可复用、失败后可否续跑。这些论文已显现「Agent OS 零部件」的模样，但还不是真正的 Agent 操作系统。

![ICML 2026 Agent Infra 论文分布](../_assets/Foundations-%E5%9F%BA%E7%A1%80%E6%A6%82%E5%BF%B5/icml26-15-agent-overview.jpg)

### 4.1 ThunderAgent：以 Agentic Program 为调度对象

[ThunderAgent: A Simple, Fast and Program-Aware Agentic Inference System](https://arxiv.org/pdf/2602.13692)（Georgia Tech、UIUC、CMU、Together AI）

**背景**：传统 Serving 的对象是一条请求（Prompt → Prefill → Decode → Response），但 Agent 任务是一串交替的模型调用与外部执行（Reasoning → Tool Call → 等待 → Reasoning → …）。现有系统各看局部：LLM 引擎看到独立 request，K8s 看到容器，Tool Runtime 看到 shell/browser/API 调用——**没有一个系统看到完整的 Agent 生命周期**。

**方案**：给每个 Agent 任务建立持续存在的 **Program object**，模型请求、工具调用和资源都归属同一程序：

- 每次 LLM call 和 tool call 携带相同 `program_id`，runtime 可跨多轮恢复完整任务关系；
- 持续记录 context size、tool environment、GPU placement、Reasoning/Acting phase、Active/Paused/Terminated 状态；
- GPU KV 压力过高时，scheduler **优先暂停正在等工具的 Program**，其他节点有空间时恢复或重放置；
- tool manager 提前准备环境，程序结束后回收磁盘、端口和 sandbox。

信息流闭环：Agent request → Program State Table → Scheduler → vLLM/SGLang/Tool 环境 → 资源状态反馈 → 更新 Program 状态进入下轮调度——从一次性 routing 变成持续管理 Agent 生命周期。

![ThunderAgent 流程](../_assets/Foundations-%E5%9F%BA%E7%A1%80%E6%A6%82%E5%BF%B5/icml26-17-thunderagent.jpg)

**意义**：把 Agent Serving 的核心对象改写为跨模型、工具、持续存在的 program；可作为 vLLM/SGLang 与工具平台之间的控制层，减少重复 prefill、跨节点显存失衡与 sandbox/磁盘泄漏。

### 4.2 CONCUR：基于拥塞控制的 Agent 并发调节

[CONCUR: High-Throughput Agentic Batch Inference of LLM via Congestion-Based Concurrency Control](https://arxiv.org/pdf/2601.22705)（南洋理工等）

**背景**：Agent 每轮 Reasoning + Acting 都会把新思考、工具结果、环境观测追加进上下文——输入长度与 KV Cache 占用**随执行步数持续增长**；且各 Agent 进度不同步（A 在生成、B 在等搜索 API、C 在等代码执行）。KV Cache 变成长期存在、动态增长、被大量 Agent 争抢的资源。传统「缓存满 → 驱逐 → Agent 回来 → 重新 Prefill」造成 **middle-phase thrashing**：显存看似能跑，GPU 却大量时间重复计算刚丢掉的历史。

**方案**：把 GPU KV Cache 看作有限共享带宽，把「活跃 Agent 数量」看作需动态控制的**拥塞窗口（congestion window）**——不要等堵住再清缓存，而要先控制放多少 Agent 进来。核心是一个 Agent 级 Admission Controller：持续读取 KV Cache 利用率和命中率，以 Agent 为对象执行 admit/pause/resume；利用率低时 **Additive Increase**（逐步加），利用率高且命中率下降时 **Multiplicative Decrease**（快速缩窗）——即网络拥塞控制的 AIMD 思想。本质是**在拥堵形成前限流**：提前降并发 → 稳住 working set → 避免重复计算。

**意义**：把 Agent KV Cache 问题重定义为「控制多少长期 Agent 可同时活跃」，缓存管理提升为 flow control 问题；作为轻量控制层插在 Agent Framework 与 Serving Engine 之间，部署改动集中，适合 Agent RL rollout、批量评测、数据蒸馏。

### 4.3 GraphFlow：基于图的 Agent Workflow 管理

[GraphFlow: A Graph-Based Workflow Management for Efficient LLM-Agent Serving](https://arxiv.org/pdf/2605.22566)（西安交通大学）

**背景**：早期 Agent workflow 是人写 SOP 或固定 Prompt Chain（稳定但只能处理预设任务）；检索式方案从库中找最相似模板复用，但 workflow 被当作**不可拆分的整体**——新任务若需要「A 的前两步 + B 的后两步」，整模板检索就难以处理。

**方案**：把所有 workflow 拆解合并为全局共享 **wGraph**：节点是原子操作、边是合法依赖，每个任务只从中选出专属子图（「流程积木库」，来任务现场拼装）。请求到来时把任务语义注入 wGraph，用图神经网络选出子图；同一张图既决定 Agent 执行哪些步骤，也决定相同步骤的 KV 状态如何共享（**Base KV + Residual KV 重建**），避免重复计算与重复存储。

流程：历史 Workflow → 原子操作抽取 → 全局 wGraph → 任务子图生成 → Base KV + Residual KV 重建 → Agent 执行。

**意义**：提出连接 Agent planning 与 serving state management 的共享 Workflow IR，workflow 从独立 Prompt/模板/DAG 演化为可组合、可查询、可复用状态的全局图；适合操作集合稳定但任务组合繁多的企业 Agent、Data Agent、Coding Agent。

## 5. 相关页面

- [vLLM 论文精读](../04-Inference-%E6%8E%A8%E7%90%86%E5%9F%BA%E7%A1%80%E8%AE%BE%E6%96%BD/vLLM%20%E8%AE%BA%E6%96%87%E7%B2%BE%E8%AF%BB.md)、[KV Cache 内存管理（PagedAttention）](../04-Inference-%E6%8E%A8%E7%90%86%E5%9F%BA%E7%A1%80%E8%AE%BE%E6%96%BD/KV%20Cache%20%E5%86%85%E5%AD%98%E7%AE%A1%E7%90%86%EF%BC%88PagedAttention%EF%BC%89.md) — Serving 脉络（Orca→vLLM→Sarathi→DistServe→Mooncake）中本站已有的两站
- [推理全流程串讲（概览篇）](../04-Inference-%E6%8E%A8%E7%90%86%E5%9F%BA%E7%A1%80%E8%AE%BE%E6%96%BD/LLM%20%E6%8E%A8%E7%90%86%E5%85%A8%E6%B5%81%E7%A8%8B%E6%A6%82%E8%A7%88.md) — Prefill/Decode、TTFT/TPOT 基础
- [通信原语与 NCCL](%E9%80%9A%E4%BF%A1%E5%8E%9F%E8%AF%AD%E4%B8%8E%20NCCL.md) — DITRON 要把通信从库调用变成编译对象，NCCL/All-to-All 是其改造的起点
- [Attention 与 GQA](../../llm/01-Architecture-%E6%A8%A1%E5%9E%8B%E6%9E%B6%E6%9E%84/Attention%20%E4%B8%8E%20GQA.md) — UPipe 的 GQA 兼容调度
- [Hy3 preview Hopper 推理优化实践](../04-Inference-%E6%8E%A8%E7%90%86%E5%9F%BA%E7%A1%80%E8%AE%BE%E6%96%BD/%E6%8E%A8%E7%90%86%E6%80%A7%E8%83%BD%E5%85%A8%E6%A0%88%E4%BC%98%E5%8C%96%E5%AE%9E%E8%B7%B5%EF%BC%88Hopper%20%E6%A1%88%E4%BE%8B%EF%BC%89.md) — 工业界对照：Quartet II 的 Hadamard 旋转与 Hy3 量化节同款技术；PPD/AMPD 讨论的 P/D 分离与 Hy3 的 Prefill/Decode 并行策略呼应
