# KV Cache 内存管理（PagedAttention）

> 概念页：以 vLLM 论文（arXiv:2309.06180）为核心参考整理，面向通用理解，后续推理框架（SGLang、TensorRT-LLM 等）可复用此页。

**KV Cache** 是自回归推理中缓存历史 token 的 Key/Value 向量的显存区，**PagedAttention** 是把操作系统分页思想用于 KV Cache 内存管理的注意力算法——它让 KV Cache 可以存放在非连续内存中,按需分配、按块共享,解决了连续内存分配导致的碎片化与浪费。

## 一、为什么需要 KV Cache

自回归生成每一步只算一个新 token,但它要与**前面所有 token** 的 K/V 向量做注意力。如果每步都重算历史 K/V,计算量随序列长度平方增长。KV Cache 把已算出的 K/V 缓存下来,每步只需:

- 计算新 token 的 $k$、$v$;
- 用新 token 的 $q$ 与缓存的所有 $k$ 做点积,对缓存的 $v$ 加权求和。

代价是**用显存换时间**:缓存随序列增长而增长,且需要在请求间共享 GPU 显存。机制级展开见 [推理全流程串讲（概览篇）](推理全流程串讲（概览篇）.md)。

## 二、KV Cache 为什么是内存瓶颈

三个特性让 KV Cache 的管理变得困难：

1. **巨大**：13B OPT 单 token 的 KV 占 800KB,单请求(2048 token)最多 1.6GB;模型越大、层越多、hidden 越大,缓存越大。
2. **动态伸缩**：随生成不断增长,生命周期=请求生命周期,长度**事先未知**。
3. **显存占比高**：A100 40GB 上约 30% 显存被 KV Cache 占用,直接决定能塞进多少个请求(batch size)。

由于服务吞吐受 batch size 限制,而 batch 受显存限制,LLM 服务本质**内存受限**。相关背景见 [AI Infra 领域概览](../02-Foundations-%E5%9F%BA%E7%A1%80%E6%A6%82%E5%BF%B5/AI%20Infra%20%E9%A2%86%E5%9F%9F%E6%A6%82%E8%A7%88.md)。

## 三、连续分配的问题

主流深度学习框架要求张量存于**连续内存**,因此早期系统为每个请求按"最大序列长度"预分配一块连续空间,产生三类浪费:

- **预留**:为未来 token 保留的空位,整段占用请求生命周期。
- **内部碎片**:实际生成长度远小于预分配长度。
- **外部碎片**:各请求预分配大小不一,分配器留下无法利用的空洞。

实测连续分配系统的有效利用率仅 **20.4%-38.2%**。压缩(compaction)理论上可缓解,但对性能敏感的推理系统不可行。

## 四、PagedAttention：虚拟内存思想

类比操作系统分页:**KV block = 页、token = 字节、request = 进程**。

- KV cache 划分为**固定大小 KV blocks**(默认 16 token/块)。
- 块可存于**非连续物理内存**,**block table**(类似页表)维护逻辑块→物理块映射及每块已填充位置数。
- 按需分配:prompt 阶段只分配实际需要的块,生成中填满一块再申请新块——**浪费被限制在一个块内**。
- 效果:消除外部碎片、缓解内部碎片、支持按块粒度共享。

PagedAttention 内核按 block table 逐块取 K/V 计算注意力,因此能处理非连续存储。

## 五、Copy-on-Write 与引用计数：共享

一个物理块可被多个逻辑块映射,因此引入**引用计数**;写一个被共享的块时触发 **copy-on-write**(COW,先复制一块再写),与 OS fork 进程一致。由此支持:

- **并行采样**:同 prompt 的多条输出共享 prompt 的 KV blocks(节省 16.2%-30.5%)。
- **Beam search**:候选束共享前缀块、随解码动态变化(类进程树,节省 44.3%-66.3%)。
- **共享前缀**:服务商预存公共 system prompt 的 KV blocks,请求直接映射(类 OS 共享库)。

统一 block table 把不同序列的共享差异隐藏起来,使系统可同时处理不同解码偏好的请求。

## 六、抢占与恢复

显存不足时按 FCFS 抢占后到请求,驱逐策略 **all-or-nothing**(序列的块整体驱逐,因整序列 KV 一起访问),恢复方式二选一:

- **Swapping**:驱逐块拷到 CPU RAM,需要时拷回;小块时 PCIe 传输开销高。
- **Recomputation**:重算被抢占序列的 KV;解码 token 与原 prompt 拼成新 prompt 一次算完,开销与块大小无关。
- 权衡:小块偏好 recomputation,大块偏好 swapping,中块(16-64)两者相当。

## 七、与相关技术的关系

| 技术 | 层级 | 解决的问题 | 关系 |
|------|------|-----------|------|
| **PagedAttention** | 内存管理层 | KV 碎片化、浪费、共享 | 本页主题 |
| **Orca 迭代级调度** | 调度层 | 请求级 batching 的排队/填充浪费 | **互补**:调度提高 GPU 利用率,PagedAttention 提高内存利用率 |
| **FlashAttention** | 算子层 | 减少注意力 I/O、峰值显存 | 可叠加:PagedAttention 管缓存存放,FlashAttention 优化计算 |
| **GQA/MQA** | 模型结构层 | 减少 K/V 头数从而减缓存 | 直接缩小 [Attention 与 GQA](../../llm/01-Architecture-%E6%A8%A1%E5%9E%8B%E6%9E%B6%E6%9E%84/Attention%20%E4%B8%8E%20GQA.md) 的缓存规模 |

完整系统设计见 [vLLM 论文精读](vLLM%20%E8%AE%BA%E6%96%87%E7%B2%BE%E8%AF%BB.md)。

## 相关笔记

- **论文来源**：[vLLM 论文精读](vLLM%20%E8%AE%BA%E6%96%87%E7%B2%BE%E8%AF%BB.md) — 系统设计、调度、实验与分布式执行的完整解析。
- **领域定位**：[AI Infra 领域概览](../02-Foundations-%E5%9F%BA%E7%A1%80%E6%A6%82%E5%BF%B5/AI%20Infra%20%E9%A2%86%E5%9F%9F%E6%A6%82%E8%A7%88.md) — 推理优化三板斧中的 KV Cache。
- **机制基础**：[推理全流程串讲（概览篇）](推理全流程串讲（概览篇）.md) — Prefill/Decode 差异化处理。
- **结构关联**：[Attention 与 GQA](../../llm/01-Architecture-%E6%A8%A1%E5%9E%8B%E6%9E%B6%E6%9E%84/Attention%20%E4%B8%8E%20GQA.md) — GQA 直接决定 KV Cache 的显存占用。
