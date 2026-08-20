# vLLM 论文精读

> 信源：Efficient Memory Management for Large Language Model Serving with PagedAttention，Woosuk Kwon 等（UC Berkeley / Stanford），SOSP 2023，arXiv:2309.06180。内容经 firecrawl scrape 抓取网页链接后提炼，非全文转载。

**vLLM** 是当前 LLM 推理基础设施的事实标准引擎。其论文的核心贡献是 **PagedAttention**——一个受操作系统**虚拟内存与分页**启发的注意力算法,把 KV cache 切分成固定大小的块、存于非连续物理内存,配合块级内存管理与抢占式调度,实现 KV cache 内存**近零浪费**,使服务吞吐较 FasterTransformer、Orca 提升 **2-4×**。

---

## 一、背景：LLM 服务为什么是内存受限的

- 自回归生成逐个 token 串行,工作负载**内存受限**(memory-bound),GPU 算力被浪费。
- 提升吞吐靠**批量处理**(batching)多个请求,但 batch 大小被 GPU 显存锁死——尤其是 **KV cache** 占用的空间。
- 以 A100 40GB 为例:模型权重约占 5%(静态),**KV cache 约占 30%**(动态,随请求分配/释放),激活只占少量。
- 趋势:GPU 算力增速快于显存容量(A100→H100 FLOPS 翻倍、显存仍 80GB),内存会越来越成为瓶颈。

## 二、问题：KV cache 内存的巨大浪费

KV cache 不同于传统张量:**动态增长/收缩、生命周期与长度事先未知**。现有系统按"每个请求的最大序列长度"预分配连续内存,产生三类浪费:

1. **预留(reserved)**:为未来 token 预留的空位,占用整个请求生命周期。
2. **内部碎片**:实际生成长度远短于最大长度(如请求最大 2048,实际只有几十)。
3. **外部碎片**:不同请求预分配大小不一,内存分配器(buddy allocator)留下无法利用的空洞。

论文实测:现有系统仅有 **20.4%-38.2%** 的 KV cache 内存用于存储真实 token 状态。更大的模型浪费更明显——13B OPT 单个 token 的 KV 占 800KB(5120 hidden × 40 层 × 2 字节),单请求最多 1.6GB。

## 三、PagedAttention：虚拟内存思想搬到注意力

类比操作系统分页:**KV block = 页、token = 字节、request = 进程**。

- 把每个序列的 KV cache 划分成**固定大小的 KV blocks**(默认每块 16 个 token)。
- KV blocks 可存储在**非连续物理内存**中,由 **block table** 维护逻辑块→物理块的映射(类似 OS 的页表)。
- 效果:
  - **消除外部碎片**(所有块同尺寸);
  - **缓解内部碎片**(小块 + 按需分配,浪费被限制在一个块内);
  - **按需动态分配**(不用预占最大长度)。

PagedAttention 内核按 block table 逐个取块计算注意力:每块内做 query 与 key 块的乘法、与 value 块加权,最终合并输出。代价是相比最优化内核(如 FasterTransformer)注意力核延迟高 20-26%,但端到端吞吐大幅胜出。

## 四、vLLM 系统设计

- **集中式 scheduler** + 分布式 GPU workers:调度器准备 input token 与每个请求的 block table,广播给 workers;workers 按 block table 读 KV cache 执行模型,GPU 间用 all-reduce 同步(无需调度器协调)。
- **KV cache manager**:在 GPU 上把连续显存切成物理块;CPU RAM 上也有一份(用于 swapping)。维护逻辑→物理映射与每块"已填充位置数"。
- 解码流程:prompt 阶段按需分配前几个块 → 自回归阶段新 token 填满最后一块后,再分配新物理块,更新 block table。
- **引用计数 + copy-on-write**:一个物理块可被多个逻辑块映射(共享),写共享块时触发 COW(复制一块再写),与 OS fork 进程的 COW 一致。

## 五、内存共享：各类解码算法

| 场景 | 共享方式 | 内存节省(实验) |
|------|----------|----------------|
| **并行采样** | 同一 prompt 的多条输出共享 prompt 的 KV cache;生成部分 COW | 16.2%-30.5%(ShareGPT) |
| **Beam search** | 候选束之间共享前缀块,随解码动态变化(类进程树) | 44.3%-66.3% |
| **共享前缀** | 服务商预存公共 system prompt 的 KV blocks,请求直接映射(类 OS 共享库) | 一shot 1.67×、few-shot 3.58× 吞吐 |
| **混合解码** | 统一 block table 掩盖不同序列的共享差异,可同时处理不同解码偏好的请求 | — |

## 六、调度与抢占

- 调度策略:**FCFS**(先来先服务),避免饥饿;内存不足时**后到的先被抢占**。
- 驱逐:**all-or-nothing**(一个序列的块要么全驱逐要么全保留,因为整序列的 KV 一起访问);同一请求内多个序列(如 beam 候选)组成 **sequence group**,一起抢占/恢复。
- 恢复方式二选一:
  - **Swapping**:把被逐块的 KV cache 拷到 CPU RAM,需要时拷回;开销取决于 CPU↔GPU 带宽,小块时效率低。
  - **Recomputation**:重算被抢占序列的 KV cache;解码 token 可与原 prompt 拼成新 prompt 一次算完,开销与块大小无关。
  - 结论:小块偏好 recomputation,大块偏好 swapping,中块(16-64)两者相当。

## 七、分布式执行

vLLM 支持 **Megatron-LM 风格张量并行**(SPMD):线性层按块切分、GPU 间 all-reduce 同步;attention 按 head 维度切分,每个 SPMD 进程负责一部分 head。

关键洞察:**模型并行下每个模型分片仍处理相同输入 token、需要相同位置的 KV cache**,因此 vLLM 采用**单一 KV cache manager**(位于集中式 scheduler),所有 workers 共享逻辑→物理映射;每个 worker 只存自己负责那部分 head 的 KV cache。

## 八、实现

- FastAPI 前端,兼容 OpenAI API;引擎 8.5K 行 Python + 2K 行 CUDA。
- 三种核心方法覆盖所有解码算法:**fork**(从既有序列派生新序列)、**append**(追加 token)、**free**(删除序列)。
- CUDA kernel 优化:融合 reshape+块写入、块读取与注意力融合(每 warp 读一块,支持变长序列)、批量块拷贝(合并 COW 的小拷贝)。

## 九、实验结论

- 基础采样:ShareGPT 下 vLLM 可比 Orca(Oracle)多扛 1.7-2.7× 请求率、比 FasterTransformer 高最多 22×(后者无细粒度调度)。
- 长序列/大模型/复杂解码场景优势更明显;短序列且显存充足的配置(如 OPT-175B + Alpaca)下差距缩小(变为算力受限)。
- Block size 权衡:太小并行度低,太大内部碎片增;默认 **16**。
- 与 Orca 互补:Orca 用迭代级调度提高 GPU 利用率,vLLM 用内存效率让更多请求放进 batch——细粒度调度反而让内存管理更关键。

## 十、影响

vLLM 已成为 LLM 推理基础设施的事实标准(开源项目 vLLM-project/vLLM),PagedAttention 及其后继(Prefix Caching、Chunked Prefill 等)支撑了主流推理框架。相关概念详见 [KV Cache 内存管理（PagedAttention）](KV%20Cache%20%E5%86%85%E5%AD%98%E7%AE%A1%E7%90%86%EF%BC%88PagedAttention%EF%BC%89.md)。

## 相关笔记

- **核心概念**：[KV Cache 内存管理（PagedAttention）](KV%20Cache%20%E5%86%85%E5%AD%98%E7%AE%A1%E7%90%86%EF%BC%88PagedAttention%EF%BC%89.md) — 分页思想、block table、copy-on-write 的通用分析。
- **领域定位**：[AI Infra 领域概览](../02-Foundations-%E5%9F%BA%E7%A1%80%E6%A6%82%E5%BF%B5/AI%20Infra%20%E9%A2%86%E5%9F%9F%E6%A6%82%E8%A7%88.md) — vLLM 作为推理 infra 代表框架、KV Cache 优化三板斧。
- **推理机制**：[推理全流程串讲（概览篇）](LLM%20%E6%8E%A8%E7%90%86%E5%85%A8%E6%B5%81%E7%A8%8B%E6%A6%82%E8%A7%88.md) — Prefill/Decode、KV Cache 的机制级展开。
- **分布式关联**：[张量并行（模型并行）](../03-Training-%E8%AE%AD%E7%BB%83%E5%9F%BA%E7%A1%80%E8%AE%BE%E6%96%BD/%E5%BC%A0%E9%87%8F%E5%B9%B6%E8%A1%8C%EF%BC%88%E6%A8%A1%E5%9E%8B%E5%B9%B6%E8%A1%8C%EF%BC%89.md) 与 [Megatron-LM 论文精读](../03-Training-%E8%AE%AD%E7%BB%83%E5%9F%BA%E7%A1%80%E8%AE%BE%E6%96%BD/Megatron-LM%20%E8%AE%BA%E6%96%87%E7%B2%BE%E8%AF%BB.md) — vLLM 分布式执行采用的张量并行策略。
