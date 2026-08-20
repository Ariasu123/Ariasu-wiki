## 学习规划

关联阅读：可结合 [MoE学习笔记](../../llm/01-Architecture-%E6%A8%A1%E5%9E%8B%E6%9E%B6%E6%9E%84/%E6%B7%B7%E5%90%88%E4%B8%93%E5%AE%B6%E6%A8%A1%E5%9E%8B%EF%BC%88MoE%EF%BC%89.md) 理解；推理优化路线可进一步分析 MoE 的部署成本。


由于大模型推理（Inference）天生受限于“内存带宽墙”（Memory Wall），且自回归（Auto-regressive）生成的特性导致计算效率低下，为了解决这些痛点的三大核心支柱：**底层 GPU 编程、模型量化压缩理论、以及系统级加速框架**。

# CUDA 编程与 GPU 架构
AI Infra 的尽头是算子手写与显存优化。不理解 GPU 架构，就无法写出极致性能的推理代码。

- **《大规模并行处理器编程实战 (PMPP)》**： GPU 架构和并行计算。理解硬件是如何执行任务的（Warp 调度、内存合并访问等）。

- **《CUDA 编程：基础与实践》 (樊哲勇)**：相比 PMPP 更加注重代码实践和快速上手，适合作为进入 CUDA 世界的第一块敲门砖。

- **[NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.cn/cuda/cuda-c-programming-guide/index.html#cuda-enabled-gpus)**：官方文档，遇到具体 API 或进阶特性（如 PTX, Tensor Cores）时的权威字典。

- **[LeetGPU](https://leetgpu.com)**：实战刷题网站。巩固 Reduce、Scan、Matrix Multiplication 等基础算子的编写。

# LLM 模型量化 (Quantization)
大模型推理通常是 Memory-bound（访存密集型）而非 Compute-bound（计算密集型）。量化是通过降低权重或激活值的精度，来大幅减少显存占用和访存带宽的核心技术。

- **GPTQ**: _Weight-Only 权重量化的代表作_。重点解决了如何高效地将模型权重压缩到 4-bit 或 3-bit，同时保持极高的精度。
- 
    - 论文：[GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)
    
- **AWQ (Activation-aware Weight Quantization)**: _端侧/边缘计算友好的量化_。它发现并非所有权重都同等重要，通过观察激活值（Activation）的分布来保护那 1% 的显著权重（Salient Weights），量化效果极好且容易在硬件上实现。
    
    - 论文：[AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978)
- **SmoothQuant**: _W8A8 (权重和激活同时量化) 的里程碑_。大模型在超过 6.7B 参数后，激活值会出现极端的异常值（Outliers），导致传统量化崩溃。SmoothQuant 通过数学平滑手段解决了这个问题，是企业级部署的标配。
    
    - 论文：[SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438)
# 算法级加速：打破自回归瓶颈与推理框架
**MEDUSA**: _推测解码（Speculative Decoding）的进化版_。传统的 LLM 一次只能吐出一个词，速度极慢。Medusa 通过在模型头部增加多个“解码头”，一次预测多个后续 Token，然后并行验证，大幅提升了生成速度，且不需要额外部署一个“小草稿模型”。

- 论文/项目：[Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774)

**TensorRT-LLM**: 它将上述的量化技术（AWQ, SmoothQuant）、算子优化（FlashAttention）、以及系统级调度（In-flight Batching）全部融合在了一起。
- 项目：[NVIDIA/TensorRT-LLM GitHub](https://github.com/NVIDIA/TensorRT-LLM)

## 相关笔记

- **延伸**：[MoE学习笔记](../../llm/01-Architecture-%E6%A8%A1%E5%9E%8B%E6%9E%B6%E6%9E%84/%E6%B7%B7%E5%90%88%E4%B8%93%E5%AE%B6%E6%A8%A1%E5%9E%8B%EF%BC%88MoE%EF%BC%89.md) — 推理优化路线可进一步分析 MoE 的部署成本。
- **实践案例**：[Hy3 preview Hopper 推理优化实践](../04-Inference-%E6%8E%A8%E7%90%86%E5%9F%BA%E7%A1%80%E8%AE%BE%E6%96%BD/%E6%8E%A8%E7%90%86%E6%80%A7%E8%83%BD%E5%85%A8%E6%A0%88%E4%BC%98%E5%8C%96%E5%AE%9E%E8%B7%B5%EF%BC%88Hopper%20%E6%A1%88%E4%BE%8B%EF%BC%89.md) — 腾讯混元在 Hopper 卡上从算子到系统的全栈推理优化，覆盖本路线多数主题的工业界落地。
- **新范式**：[Agent 驱动 GPU Kernel 生成（MLSys 2026 FlashInfer 比赛）](../04-Inference-%E6%8E%A8%E7%90%86%E5%9F%BA%E7%A1%80%E8%AE%BE%E6%96%BD/Agent%20%E9%A9%B1%E5%8A%A8%E7%9A%84%20GPU%20Kernel%20%E7%94%9F%E6%88%90%E4%B8%8E%E4%BC%98%E5%8C%96.md) — 用代码智能体自动生成与优化 kernel：Harness 闭环 + 演化搜索。
- **外部资料**：[LLM Infra 推理学习资料清单](LLM%20Infra%20%E6%8E%A8%E7%90%86%E5%AD%A6%E4%B9%A0%E8%B5%84%E6%96%99%E6%B8%85%E5%8D%95.md) — 业界工程师整理的八模块资料合集（CUDA/kernel、vLLM/SGLang、性能分析、RL、课程与实验室）。
- **模型侧**：[LLM预训练底层机制](../../llm/02-Pretraining-%E9%A2%84%E8%AE%AD%E7%BB%83/LLM%20%E9%A2%84%E8%AE%AD%E7%BB%83%E5%BA%95%E5%B1%82%E6%9C%BA%E5%88%B6.md) — Infra 路线中的 GPU 与算子知识支撑训练系统。
- **系统视角**：[LLM 综合复习](../../llm/05-Review-%E5%A4%8D%E4%B9%A0%E6%80%BB%E7%BB%93/LLM%20%E7%BB%BC%E5%90%88%E5%A4%8D%E4%B9%A0%2022%20%E9%97%AE.md) — Infra 路线把 LLM 原理延伸到性能与部署。
