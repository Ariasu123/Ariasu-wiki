
> 来源：知乎专栏《[基于代码智能体的 GPU Kernel 生成与优化：MLSys 2026 FlashInfer 比赛总结](https://zhuanlan.zhihu.com/p/2045305961280230498)》（作者：yue shui，百度）。团队在 MLSys 2026 NVIDIA Track: FlashInfer AI Kernel Generation Contest 中 Track A（Fused MoE）与 Track C（Gated Delta Net）获第 3 名、Track C Full-Agent 获第 2 名。代码与报告公开于 [mlsys26-flashinfer-contest](https://github.com/syhya/mlsys26-flashinfer-contest)。

## 目录

- [1. 研究背景](#1-研究背景)
- [2. 两套方案：Agent-Assisted vs Full-Agent](#2-两套方案agent-assisted-vs-full-agent)
- [3. Agent-Assisted：Harness Engineering 闭环](#3-agent-assistedharness-engineering-闭环)
- [4. Full-Agent：LoongFlow 演化搜索](#4-full-agentloongflow-演化搜索)
- [5. 实验结果](#5-实验结果)
- [6. 优化轨迹分析](#6-优化轨迹分析)
- [7. 未来工作](#7-未来工作)
- [8. 参考文献（节选）](#8-参考文献节选)
- [9. 相关页面](#9-相关页面)

---

## 1. 研究背景

让 LLM 生成 GPU kernel 的评测与相关工作：

- **KernelBench**（[arXiv:2502.10517](https://arxiv.org/abs/2502.10517)）：LLM 读取 PyTorch 参考实现、生成自定义内核，同时评估编译、正确性和运行性能。
- **FlashInfer-Bench**（[arXiv:2601.00227](https://arxiv.org/abs/2601.00227)）：把 kernel 生成放进大模型推理服务的真实负载分布中，强调执行轨迹、评测、候选实现和部署之间的闭环。

![KernelBench 评测流程](../_assets/Inference-%E6%8E%A8%E7%90%86%E5%9F%BA%E7%A1%80%E8%AE%BE%E6%96%BD/flashinfer-contest-fig01.jpg)

![FlashInfer-Bench 架构](../_assets/Inference-%E6%8E%A8%E7%90%86%E5%9F%BA%E7%A1%80%E8%AE%BE%E6%96%BD/flashinfer-contest-fig02.jpg)

方法分类（参考综述 [Towards Automated Kernel Generation in the Era of LLMs](https://arxiv.org/abs/2601.15727)）：

- **LLM4Kernel**：基于高质量领域语料，结合 CPT/SFT/RL 把内核知识内化到模型参数。优势是知识进参数，代价是需要高质量训练数据、稳定奖励设计和较高训练成本。
- **Agent4Kernel**：强调迭代搜索、外部记忆、硬件 profiling 与多智能体编排，与 Meta 的 **KernelEvolve**（[arXiv:2512.23236](https://arxiv.org/abs/2512.23236)，持久化知识库 + 检索增强提示 + 跨硬件抽象 + 生产算子评估）相呼应。

![LLM4Kernel 路线](../_assets/Inference-%E6%8E%A8%E7%90%86%E5%9F%BA%E7%A1%80%E8%AE%BE%E6%96%BD/flashinfer-contest-fig03.jpg)

![Agent4Kernel 路线](../_assets/Inference-%E6%8E%A8%E7%90%86%E5%9F%BA%E7%A1%80%E8%AE%BE%E6%96%BD/flashinfer-contest-fig04.jpg)

GPU 内核智能体的核心难点：在复杂硬件、复杂负载和严格验证约束下**保留有效经验**，并把失败反馈压缩成下一轮可执行的搜索约束。

## 2. 两套方案：Agent-Assisted vs Full-Agent

团队提交了 Agent-Assisted 和 Full-Agent 两套路线。**区别不在于是否使用 LLM，而在于人类是否持续介入搜索过程**。

| 维度 | Agent-Assisted | Full-Agent |
| --- | --- | --- |
| 人类介入 | 人类持续设计策略、提供参考实现、筛选优化方向、维护晋级规则 | 人类只提供初始任务、约束和自动化工具，后续由智能体自主搜索 |
| 搜索方式 | 按性能剖析结果和经验选择候选实现家族，优先做有把握的局部优化 | 按「规划、执行、评测、总结、存储」循环自动展开长程搜索 |
| 状态管理 | 人工笔记、技能文件和实验产物归档 | LoongFlow 数据库记录候选来源、结果摘要、当前最优和失败模式 |

## 3. Agent-Assisted：Harness Engineering 闭环

**Harness Engineering** 范式：人类负责设计约束、构建反馈机制并定义评估标准，Agent 在受控环境内迭代生成更高质量的代码。

![Agent-Assisted 闭环 harness](../_assets/Inference-%E6%8E%A8%E7%90%86%E5%9F%BA%E7%A1%80%E8%AE%BE%E6%96%BD/flashinfer-contest-fig06.jpg)

Harness 分四层：

1. **Grounding inputs**：算子定义、参考实现和 workload JSON 等必要上下文输入；
2. **Shape discovery**：从 workload 按 batch size、sequence length 等参数分组，每组抽取代表性维度，使 agent 可快速评估迭代候选而不必每次全量验证；
3. **Closed-loop optimization**：在 `baseline → profile → diagnose → generate → evaluate → archive` 循环中生成候选，验证编译、正确性与性能，并用 Torch Profiler 和 NVIDIA Nsight Compute（NCU）分析瓶颈；
4. **Outputs**：归档代码与性能指标，供 agent 后续迭代。

人类编写优化 [skills](https://github.com/syhya/mlsys26-flashinfer-contest/tree/main/agent-assisted/skills)、构建[评测脚手架](https://github.com/syhya/mlsys26-flashinfer-contest/tree/main/agent-assisted/scripts)——与 Agent Skills、Subagents 在上下文复用、工具封装和并行探索上的工程动机一致，把搜索过程约束在可验证的闭环里。

## 4. Full-Agent：LoongFlow 演化搜索

使用 **LoongFlow**（[arXiv:2512.24077](https://arxiv.org/abs/2512.24077)）的计划-执行-总结范式（类似 [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve) 这类演化搜索系统）：一次内核搜索拆成**规划、执行、评测、总结、存储**五步，每个候选的来源、性能结果和失败原因摘要写入持久数据库，失败候选转化为后续迭代可检索的上下文。

![LoongFlow Full-Agent 栈](../_assets/Inference-%E6%8E%A8%E7%90%86%E5%9F%BA%E7%A1%80%E8%AE%BE%E6%96%BD/flashinfer-contest-fig07.jpg)

## 5. 实验结果

官方 leaderboard Top-3：Track A Fused MoE（Agent-Assisted，3rd）、Track C Gated Delta Net（Agent-Assisted，3rd；Full-Agent，2nd）。

加速比定义：

$$\mathrm{Speedup} = \frac{\text{mean baseline latency}}{\text{mean solution latency}}$$

下表为本地 Modal B200 评测（时钟频率无法完全锁定，仅作参考），用于分析保留产物的优化幅度，非官方逐算子 contest score：

| 算子定义 | 方法 | 平均延迟 (ms) | 相对 PyTorch reference 加速 | 相对 FlashInfer baseline 加速 |
| --- | --- | --- | --- | --- |
| DSA Attention | Agent-Assisted | 0.011175 | 217.17× | 29.68× |
| | Full-Agent | 0.022811 | 106.39× | 14.54× |
| | FlashInfer baseline | 0.331650 | 7.32× | 1.00× |
| DSA Indexer | Agent-Assisted | 0.006893 | 494.13× | 18.05× |
| | Full-Agent | 0.032659 | 104.29× | 3.81× |
| | FlashInfer baseline | 0.124420 | 27.38× | 1.00× |
| GDN Prefill | Agent-Assisted | 0.051992 | 21,078× | 13.70× |
| | Full-Agent | 0.688875 | 1,591× | 1.03× |
| | FlashInfer baseline | 0.712166 | 1,539× | 1.00× |
| MoE FP8 | Agent-Assisted | 0.286340 | 63.78× | 1.62× |
| | FlashInfer baseline | 0.463874 | 39.37× | 1.00× |
| | Full-Agent | 1.742630 | 10.48× | 0.27× |
| GDN Decode | Agent-Assisted | 0.006201 | 7,970× | 1.12× |
| | FlashInfer baseline | 0.006940 | 7,121× | 1.00× |
| | Full-Agent | 0.008366 | 5,907× | 0.83× |

![最终保留加速比](../_assets/Inference-%E6%8E%A8%E7%90%86%E5%9F%BA%E7%A1%80%E8%AE%BE%E6%96%BD/flashinfer-contest-fig08.jpg)

**Agent-Assisted 在五个算子上全部优于 FlashInfer baseline**（1.12× ~ 29.68×）；Full-Agent 在 DSA Attention、DSA Indexer、GDN Prefill 上找到有效候选，但 MoE FP8 与 GDN Decode 低于 baseline。

## 6. 优化轨迹分析

![Agent-Assisted 优化轨迹](../_assets/Inference-%E6%8E%A8%E7%90%86%E5%9F%BA%E7%A1%80%E8%AE%BE%E6%96%BD/flashinfer-contest-fig09.jpg)

- 性能提升**不是平滑发生的，而是长期平台期之后出现少数几次明显跃迁**；
- 有效的 Agent-Assisted kernel 优化不是单纯依赖提示词，而是依赖可测量的系统闭环：把算子约束、评测脚手架、性能分析反馈和历史轨迹组织成可复用流程，再让智能体在其中生成、验证和保留候选——这个过程需要人类持续设计和维护 harness。

![Full-Agent 优化轨迹](../_assets/Inference-%E6%8E%A8%E7%90%86%E5%9F%BA%E7%A1%80%E8%AE%BE%E6%96%BD/flashinfer-contest-fig10.jpg)

- Full-Agent（灰点为通过正确性的候选，实线为 running best，虚线为 baseline）也能在部分算子找到有效候选（DSA Attention 14.54×），但整体慢于 Agent-Assisted，两个算子甚至低于 baseline；
- 差距说明**完全自动化的智能体搜索仍然困难**：人类提供的高质量参考实现和持续积累的轨迹记忆，往往比让智能体从零探索更有效率。未来系统需要把控制状态和历史记忆纳入 harness，同时保持严格的最终验证。

## 7. 未来工作

- **模型级优化闭环**：结合 AutoKernel（[arXiv:2603.21331](https://arxiv.org/abs/2603.21331)），把单算子优化扩展到模型级 `profile → extract → optimize → verify` 流程：先用剖析器找 GPU 瓶颈，抽取独立 Triton/CUDA 内核，用 Amdahl 定律决定下一轮优先优化哪个内核；
- **实验管理与独立验证器**：参考官方[比赛 writeups](https://github.com/flashinfer-ai/mlsys26-contest/tree/main/writeups)，脚手架需固定基准测试、正确性检查、输入形状扫描、数值稳定性、确定性检查、Roofline 分析、保留/回滚决策和产物结构约束，并用独立验证器复核候选；
- **负载特化与可检索记忆**：高分方案的共同点不是盲目试更多 kernel，而是**先理解负载分布再选实现策略**。未来把负载画像、可复用优化模板、成功候选和失败原因结构化记录，生成代码前先检索相似场景，明确适用输入形状、已知瓶颈和不应重复尝试的方向。

## 8. 参考文献（节选）

1. [FlashInfer AI Kernel Generation Contest](https://mlsys26.flashinfer.ai/)（MLSys 2026, NVIDIA Track）
2. Shui et al. [Harness Engineering for LLM-Driven GPU Kernel Generation](https://github.com/syhya/mlsys26-flashinfer-contest/blob/main/agent-assisted/report.pdf)（2026）
3. Ma et al. [Full-Agent Kernel Generation for FlashInfer @ MLSys 2026](https://github.com/syhya/mlsys26-flashinfer-contest/blob/main/full-agent/FULL_AGENT_WRITEUP.pdf)（2026）
4. Ouyang et al. [KernelBench: Can LLMs Write Efficient GPU Kernels?](https://arxiv.org/abs/2502.10517)（2025）
5. Xing et al. [FlashInfer-Bench: Building the Virtuous Cycle for AI-driven LLM Systems](https://arxiv.org/abs/2601.00227)（2026）
6. Yu et al. [Towards Automated Kernel Generation in the Era of LLMs](https://arxiv.org/abs/2601.15727)（2026）
7. Liao et al. [KernelEvolve: Scaling Agentic Kernel Coding for Heterogeneous AI Accelerators at Meta](https://arxiv.org/abs/2512.23236)（2025）
8. Ye et al. [FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving](https://arxiv.org/abs/2501.01005)（MLSys 2025）

## 9. 相关页面

- [Harness Engineering](../../Agent/03-Harness-and-Workflows-%E8%BF%90%E8%A1%8C%E6%A1%86%E6%9E%B6%E4%B8%8E%E5%B7%A5%E4%BD%9C%E6%B5%81/Harness%20Engineering.md) — Agent-Assisted 路线即该范式在 kernel 生成比赛中的落地：人设计约束与评估，Agent 在受控闭环内迭代
- [LLM 推理加速与算子优化学习路线](../01-Roadmaps-%E5%AD%A6%E4%B9%A0%E8%B7%AF%E7%BA%BF/LLM%20%E6%8E%A8%E7%90%86%E5%8A%A0%E9%80%9F%E4%B8%8E%E7%AE%97%E5%AD%90%E4%BC%98%E5%8C%96%E5%AD%A6%E4%B9%A0%E8%B7%AF%E7%BA%BF.md) — kernel 优化是学习路线的核心主题，本文展示「让 agent 写 kernel」的新范式
- [Hy3 preview Hopper 推理优化实践](%E6%8E%A8%E7%90%86%E6%80%A7%E8%83%BD%E5%85%A8%E6%A0%88%E4%BC%98%E5%8C%96%E5%AE%9E%E8%B7%B5%EF%BC%88Hopper%20%E6%A1%88%E4%BE%8B%EF%BC%89.md) — 对照：腾讯混元人工打造 HPC-Ops 算子库 vs 本文用 agent 自动生成/优化 kernel
- [推理全流程串讲（概览篇）](LLM%20%E6%8E%A8%E7%90%86%E5%85%A8%E6%B5%81%E7%A8%8B%E6%A6%82%E8%A7%88.md) — MoE、Attention 等待优化算子在推理链路中的位置
