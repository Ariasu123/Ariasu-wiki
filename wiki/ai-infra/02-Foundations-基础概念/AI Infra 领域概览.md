
> 信源：锦恢《AI Infra 软核教程（一）为什么我们需要 AI Infra》，[知乎原文](https://zhuanlan.zhihu.com/p/2063938554061967779)（2026-07-24）。本页为该文的网状知识提炼，非全文转载。

**AI Infra（AI 基础设施）** 覆盖一切与 AI 训练、推理、评测相关的底层工程。它的核心命题是：**当模型已经无法由单张 GPU 独立完成计算之后，怎样让成百上千张 GPU 像一台机器一样协同工作？** AI Infra 的价值不在于拥有多少 GPU 和数据，而在于如何高效地调度它们。

---

## 一、AI Infra 的五大板块

| 板块 | 内容 | 代表技术/岗位 |
|------|------|--------------|
| 计算集群 | GPU/TPU 节点、通信协议、底层调度 | NCCL、Slurm、Ray；大公司里称"机架"工程师，负责网络拓扑、散热、供电，岗位极少 |
| 数据与存储（data infra） | 向量数据库、数据集/benchmark 管理、权重管理、智能体数据回流 | 随可训练数据枯竭和 harness 兴起而越来越重要 |
| 训练 infra | 让小规模训练代码在千卡/万卡集群上正常、高效、稳定运行 | [Megatron-LM](../03-Training-%E8%AE%AD%E7%BB%83%E5%9F%BA%E7%A1%80%E8%AE%BE%E6%96%BD/Megatron-LM%20%E8%AE%BA%E6%96%87%E7%B2%BE%E8%AF%BB.md)、DeepSpeed、ColossalAI；需求量最高的岗位之一 |
| 推理 infra | 模型部署上线、高可用高并发、算子融合、降低推理成本 | [vLLM](../04-Inference-%E6%8E%A8%E7%90%86%E5%9F%BA%E7%A1%80%E8%AE%BE%E6%96%BD/vLLM%20%E8%AE%BA%E6%96%87%E7%B2%BE%E8%AF%BB.md)、SGLang；同为需求量最高的岗位之一 |
| 智能体（Agent Infra） | 高性能 sandbox、Agent 数据回流、Agent 框架与评估平台、内部 MCP 服务 | OpenSandbox、LangChain、LangSmith 等；暂未独立成工种 |

两个重要概念：

- **训练后端 / 推理后端**：Agentic RL 兴起后，Megatron、DeepSpeed 这类"传统"训练框架被称为训练后端（如智谱 slime 默认用 [Megatron](../03-Training-%E8%AE%AD%E7%BB%83%E5%9F%BA%E7%A1%80%E8%AE%BE%E6%96%BD/Megatron-LM%20%E8%AE%BA%E6%96%87%E7%B2%BE%E8%AF%BB.md)）；推理侧则直接复用现成推理框架作推理后端（如 slime 默认用 SGLang）。Agentic RL 框架代表：智谱 slime、阿里 dressage、字节 verl。
- **工种界定**：业界一般只把做训练和推理的工程师称为 AI Infra 工程师（"训推"）；计算集群交由机架工程师/运维团队；data infra 与 agent infra 只在少数前沿公司划入 infra。

## 二、为什么需要 AI Infra：显存墙

大模型与传统深度神经网络在数学层面没有差别，过去却很少谈 Infra。根本原因在于**硬件**：

- 传统视觉模型参数量约 100MB，一张消费级显卡甚至树莓派即可运行。
- 主流大模型参数已达 2T 级别，FP16 下权重文件约 **4TB**，比前者高出四个数量级。
- 以 NVIDIA B200（单卡显存约 180GB）为例，一台 DGX 服务器 8 卡的总装载量为：

$$8 \times 180\mathrm{GB} = 1440\mathrm{GB} = 1.44\mathrm{TB} < 4\mathrm{TB}$$

- 模型量化（用更低精度表示部分参数）可以缓解，但不能根治；**训练所需显存约为模型大小的 8 倍**（权重 + 梯度 + 优化器状态 + 激活值），单机更不可能。

结论：单机不够就得多机——大模型训推正式迈入**分布式计算**，与高性能计算（HPC）高度合流（早期 infra 人才多出身 HPC 实验室）。多机多卡引入通信、同步、一致性等工程问题，这正是 AI Infra 工程师要回答的问题。

## 三、训练 infra 的工程难点

传统训练流程：`数据 → 前向传播 → 计算损失 → 反向传播 → 更新参数`。模型变大后平摊到多机多卡，引入的麻烦：

- **一卡一进程**：主流框架为每张 GPU 启动独立 Python 进程。进程内存彼此隔离，原本一块内存里的数据传递变成进程间、GPU 间、服务器间的通信。真正执行矩阵运算和通信的是底层 C++、CUDA、NCCL。
- **通信原语**：HPC 在九十年代就确立了**集合通信**（Collective，如 AllReduce）与**点对点通信**（P2P，如 Send/Recv）两类标准；NVIDIA 的 NCCL、华为的 HCCL 是具体实现。
- **分布式计算图**：每个进程只维护本地计算图，跨设备依赖需要额外传递激活值、梯度、参数状态——因此分布式训练的总显存占用**大于**模型本身参数。
- **并行算法是核心**：目标是在满足训练的前提下拉高各 GPU 利用率。经典方案如[张量并行 × 流水线并行组合（TP × PP）](../03-Training-%E8%AE%AD%E7%BB%83%E5%9F%BA%E7%A1%80%E8%AE%BE%E6%96%BD/%E5%BC%A0%E9%87%8F%E5%B9%B6%E8%A1%8C%EF%BC%88%E6%A8%A1%E5%9E%8B%E5%B9%B6%E8%A1%8C%EF%BC%89.md)，以及 DP、SP、CP、EP 等。
- **重要 insight**：大规模分布式训练中 GPU 利用率一般不会到 90%，优秀并行算法能让总和平均利用率达 40%+ 就已很棒。

## 四、推理 infra 的工程难点

推理无需反向传播，优化空间更大。大模型推理相比传统推理有三个特点：

1. **自回归生成**：逐个 token 串行生成（生成结果拼回输入再生成下一个，遇到 EOS 标记停止），单个 query 无法并行加速。
2. **高并发**：模型太大，普通用户只能用云服务，服务必须承受高并发。
3. **商业模式约束**：API 按量计费，推理 infra 必须保证不赔本。

由自回归特性引出的优化三板斧：

- **KV Cache**：相邻两步生成的前缀完全相同，缓存历史 K/V 避免重复计算。详见 [主题一：推理全流程串讲（概览篇）](../../llm/04-Engineering-%E5%B7%A5%E7%A8%8B%E5%AE%9E%E8%B7%B5/%E4%B8%BB%E9%A2%98%E4%B8%80%EF%BC%9A%E6%8E%A8%E7%90%86%E5%85%A8%E6%B5%81%E7%A8%8B%E4%B8%B2%E8%AE%B2%EF%BC%88%E6%A6%82%E8%A7%88%E7%AF%87%EF%BC%89.md)（含 Prefill/Decode 差异化处理、PagedAttention），分页式内存管理详见 [KV Cache 内存管理（PagedAttention）](../04-Inference-%E6%8E%A8%E7%90%86%E5%9F%BA%E7%A1%80%E8%AE%BE%E6%96%BD/KV%20Cache%20%E5%86%85%E5%AD%98%E7%AE%A1%E7%90%86%EF%BC%88PagedAttention%EF%BC%89.md)。
- **Draft Model（草稿模型/投机采样）**：一次生成多个 token 再校验，打破逐 token 串行。
- **量化压缩、算子融合**：压榨单个 transformer 模块的性能。详见 [LLM 推理加速与算子优化学习路线](../01-Roadmaps-%E5%AD%A6%E4%B9%A0%E8%B7%AF%E7%BA%BF/LLM%20%E6%8E%A8%E7%90%86%E5%8A%A0%E9%80%9F%E4%B8%8E%E7%AE%97%E5%AD%90%E4%BC%98%E5%8C%96%E5%AD%A6%E4%B9%A0%E8%B7%AF%E7%BA%BF.md)。

工业场景下还要同时服务成百上千人，在榨干服务器算力与满足业务要求之间取得平衡——这一层称为**推理中台**。

## 五、怎么做 infra（入行建议）

- **计算机基本功**：操作系统、高性能计算、计算机体系结构、芯片设计。典型案例：DeepSeek 不停留在 CUDA 抽象层，部分场景直接用 PTX 指令优化，甚至在 DeepEP 通信库中使用官方文档未公开的 PTX 指令提升 MoE 通信与推理效率，大幅压低推理成本。
- **深度学习基础**：infra 的工作对象是深度学习模型，需要模型训练经验。
- **读优秀开源项目**：Megatron-LM、DeepSpeed、SGLang、vLLM。**先读官方文档建立认知，再读源码**，快速减少 Unknown Unknown。
- 多机多卡环境多数人只能在公司/实验室获得，因此先建立原理认知，再在真实场景中用系统认知 + AI 工具解决问题。

相关学习路径见 [vLLM / SGLang 学习路线](../01-Roadmaps-%E5%AD%A6%E4%B9%A0%E8%B7%AF%E7%BA%BF/%E8%B7%AF%E7%BA%BF.md) 与 [学习规划](../01-Roadmaps-%E5%AD%A6%E4%B9%A0%E8%B7%AF%E7%BA%BF/%E5%AD%A6%E4%B9%A0%E8%A7%84%E5%88%92.md)。

## 六、原系列 roadmap（锦恢的软核教程规划）

1. 第一篇（本文）：领域概况
2. NCCL 通信原语与常见集群（NVIDIA、AWS）
3-4. 训练 infra：PP、DP、SP、CP、EP 等并行算法
5-6. 推理 infra：部署框架、草稿模型、量化技术
7+. Agent 基建：Sandbox 等 Agentic AI 组件

## 相关笔记

- **训练并行**：[Megatron-LM 论文精读](../03-Training-%E8%AE%AD%E7%BB%83%E5%9F%BA%E7%A1%80%E8%AE%BE%E6%96%BD/Megatron-LM%20%E8%AE%BA%E6%96%87%E7%B2%BE%E8%AF%BB.md) — 训练 infra 代表框架的论文级解析：层内模型并行、8.3B 训练、BERT LayerNorm 重排。
- **张量并行**：[张量并行（模型并行）](../03-Training-%E8%AE%AD%E7%BB%83%E5%9F%BA%E7%A1%80%E8%AE%BE%E6%96%BD/%E5%BC%A0%E9%87%8F%E5%B9%B6%E8%A1%8C%EF%BC%88%E6%A8%A1%E5%9E%8B%E5%B9%B6%E8%A1%8C%EF%BC%89.md) — 文中提到的 TP 并行范式的通用切分策略与通信开销分析。
- **推理详解**：[主题一：推理全流程串讲（概览篇）](../../llm/04-Engineering-%E5%B7%A5%E7%A8%8B%E5%AE%9E%E8%B7%B5/%E4%B8%BB%E9%A2%98%E4%B8%80%EF%BC%9A%E6%8E%A8%E7%90%86%E5%85%A8%E6%B5%81%E7%A8%8B%E4%B8%B2%E8%AE%B2%EF%BC%88%E6%A6%82%E8%A7%88%E7%AF%87%EF%BC%89.md) — KV Cache、EOS、Prefill/Decode 的机制级展开。
- **推理框架**：[vLLM / SGLang 学习路线](../01-Roadmaps-%E5%AD%A6%E4%B9%A0%E8%B7%AF%E7%BA%BF/%E8%B7%AF%E7%BA%BF.md) — 推理 infra 两大主流框架的上手路径。
- **推理优化**：[LLM 推理加速与算子优化学习路线](../01-Roadmaps-%E5%AD%A6%E4%B9%A0%E8%B7%AF%E7%BA%BF/LLM%20%E6%8E%A8%E7%90%86%E5%8A%A0%E9%80%9F%E4%B8%8E%E7%AE%97%E5%AD%90%E4%BC%98%E5%8C%96%E5%AD%A6%E4%B9%A0%E8%B7%AF%E7%BA%BF.md) — 量化、算子融合、CUDA 编程。
- **学习路径**：[学习规划](../01-Roadmaps-%E5%AD%A6%E4%B9%A0%E8%B7%AF%E7%BA%BF/%E5%AD%A6%E4%B9%A0%E8%A7%84%E5%88%92.md) — 8 周分阶段的 infra 学习执行计划。
- **MoE 关联**：[MoE 学习笔记](../../llm/01-Architecture-%E6%A8%A1%E5%9E%8B%E6%9E%B6%E6%9E%84/MoE%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0.md) — EP（专家并行）与 DeepEP 通信库的应用背景。
- **KV Cache 结构基础**：[Attention 与 GQA](../../llm/01-Architecture-%E6%A8%A1%E5%9E%8B%E6%9E%B6%E6%9E%84/Attention%20%E4%B8%8E%20GQA.md) — GQA 直接决定 KV Cache 的显存占用。
- **推理引擎**：[vLLM 论文精读](../04-Inference-%E6%8E%A8%E7%90%86%E5%9F%BA%E7%A1%80%E8%AE%BE%E6%96%BD/vLLM%20%E8%AE%BA%E6%96%87%E7%B2%BE%E8%AF%BB.md) — PagedAttention、块级内存管理与 2-4× 吞吐提升。
- **KV 内存**：[KV Cache 内存管理（PagedAttention）](../04-Inference-%E6%8E%A8%E7%90%86%E5%9F%BA%E7%A1%80%E8%AE%BE%E6%96%BD/KV%20Cache%20%E5%86%85%E5%AD%98%E7%AE%A1%E7%90%86%EF%BC%88PagedAttention%EF%BC%89.md) — 分页思想、block table 与 copy-on-write 共享。
- **Agent 评估平台**：[Langfuse](../../agent-engineering/03-Harness-and-Workflows-%E8%BF%90%E8%A1%8C%E6%A1%86%E6%9E%B6%E4%B8%8E%E5%B7%A5%E4%BD%9C%E6%B5%81/Langfuse.md) — 文中提到的 Agent Infra 组件之一。
- **Agent 框架**：[AgentScope](../../agent-engineering/04-References-%E9%A1%B9%E7%9B%AE%E5%8F%82%E8%80%83/AgentScope.md) — 文中提到的 Agent 框架代表。
