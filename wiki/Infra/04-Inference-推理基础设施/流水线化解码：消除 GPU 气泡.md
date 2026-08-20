
> 来源：Moondream 工程博客 [Popping the GPU Bubble](https://moondream.ai/blog/popping-the-gpu-bubble)（2026-06-04）。Photon 是 Moondream 的推理引擎（NVIDIA B200 上约 33ms 的近实时 VLM 推理），本文是其流水线化解码（pipelined decoding）技术的结构化笔记，decode 吞吐最高提升 35%。

## 目录

- [1. 问题：GPU 气泡](#1-问题gpu-气泡)
- [2. 核心思路：流水线化解码](#2-核心思路流水线化解码)
- [3. 机制一：Ping-pong slots](#3-机制一ping-pong-slots)
- [4. 机制二：Forward now, sample later](#4-机制二forward-now-sample-later)
- [5. 机制三：Zombies——早收尾、晚释放](#5-机制三zombies早收尾晚释放)
- [6. Prefill 复用同一条流水线](#6-prefill-复用同一条流水线)
- [7. 气泡的成本模型与实测](#7-气泡的成本模型与实测)
- [8. 要点总结](#8-要点总结)
- [9. 相关页面](#9-相关页面)

---

## 1. 问题：GPU 气泡

自回归生成逐个产出 token，每步 decode 是 CPU 与 GPU 之间的一次往返：GPU 跑模型 forward（数十亿次运算），但 CPU 也有不少固定工作——选择下一批请求、准备 GPU 所需元数据、从输出中取出 token 并记录等。

矛盾在于：**一个 token 的 GPU 计算量很小，而 CPU housekeeping 是每步必付的固定成本**。GPU 若必须等 CPU 干完才能开始下一个 token，每轮循环都有一段空闲——这就是 **GPU 气泡（GPU bubble）**。

## 2. 核心思路：流水线化解码

![阻塞 vs 流水线 decode 时间线](../_assets/Inference-%E6%8E%A8%E7%90%86%E5%9F%BA%E7%A1%80%E8%AE%BE%E6%96%BD/photon-timeline-comparison.svg)

- **阻塞版**（上）：每步都是接力棒传递——CPU 规划并 launch forward → GPU 执行 → CPU 同步等结果、提交 → 再规划下一步。因为下一步计划依赖刚采样的 token（如模型回答完毕要调度新请求），GPU 在 CPU 的 commit-plan-launch 期间闲置。
- **流水线版**（下）：上一步 token 还在回传提交时，就启动下一步 forward，forward 首尾相接，CPU 工作被压在下面重叠执行。

**为什么可行**：刚采样的 token 不必离开 GPU——下一步 forward 直接从显存读它作输入。CPU 侧终究需要一份拷贝（detokenize、流式输出、判断请求是否结束），但那是可以稍后后台完成的簿记工作。**不等这份拷贝，就是消掉气泡的关键一步**。

安全实现需要三个机制：ping-pong slots（防止步骤间缓冲冲突）、forward now/sample later（受约束解码的采样顺序）、zombies（请求结束后的清理）。

## 3. 机制一：Ping-pong slots

一个 decode 步骤需要一组工作缓冲：输入暂存（上一步 token 及其位置）、logits 输出（词表大小）、采样结果落点、attention kernel 所需的 KV cache 簿记元数据。两端都用 **pinned（页锁定）host buffer**，使 H2D/D2H 拷贝以后台 DMA 进行而不阻塞 CPU。

关键工程约束：

- 缓冲**一次分配、每步复用**：运行期 GPU 内存分配可能引发设备同步产生气泡；固定地址也是将 decode 步骤捕获为 **CUDA Graph** 重放、降低 kernel launch 开销的前提。这一整包称为 `DecodeSlot`。
- 缓冲在步骤完成前一直被占用，会阻塞流水线——下一步若复用同一套缓冲，会在 CPU 读完前覆盖上一步结果。因此**准备两份 slot，乒乓交替**。

![Ping-pong slots](../_assets/Inference-%E6%8E%A8%E7%90%86%E5%9F%BA%E7%A1%80%E8%AE%BE%E6%96%BD/photon-pingpong-slots.svg)

关于 stream 的两个要点：

- CPU 的 launch 不是立即执行，而是 enqueue 到 **stream**（GPU 按序排空的有序队列）。同一 stream 顺序执行，不同 stream 可重叠。**两份 slot 的 forward 放同一个 compute stream**——slot 不是为了 GPU 并行，只是为了让 CPU 能处理一份 slot 的结果、同时 GPU 跑另一份的 forward。
- 每步的 D2H 拷贝（把采样 token 带回 CPU 簿记）走**独立 copy stream**，并锚定到「该步输出写完成」时记录的 event 上，从而与下一个 forward 并行——这就是「不必等拷贝」的实现方式。

![拷贝在后台进行](../_assets/Inference-%E6%8E%A8%E7%90%86%E5%9F%BA%E7%A1%80%E8%AE%BE%E6%96%BD/photon-deferred-copy.svg)

**释放时机**：slot 不是在 GPU 用完后就空闲，而是要等结果被 CPU 读完——pinned host buffer 是可能仍在飞行中的拷贝的落点，过早交给新步骤会在传输中途覆写数据（极难调试的 corruption bug）。slot 保留至读取它的 commit 完成才释放。

## 4. 机制二：Forward now, sample later

下一步 forward 不依赖 CPU 对上一步 token 的任何处理，但**下一步的采样依赖**。来源是**受约束解码（constrained decoding）**：Moondream 的空间技能返回结构化输出而非自由文本（`point` 返回坐标、`detect` 返回检测框、`segment` 返回轮廓），实现方式是每步把不允许的 token 的 logits 压为 −∞。哪些 token 允许（mask）取决于已产出的内容，因此 t+1 的 mask 依赖 t 步采样的 token。

**依赖在采样，而不在 forward。**

![forward 不需要 mask，只有采样需要](../_assets/Inference-%E6%8E%A8%E7%90%86%E5%9F%BA%E7%A1%80%E8%AE%BE%E6%96%BD/photon-advance-tick.svg)

每个调度 tick 分三相：

1. **Launch** t+1 的 forward——不依赖 mask，立即发出；
2. **Commit** 步骤 t——等待在途拷贝、推进请求解码状态（这是算 t+1 mask 的前提）；
3. **Finalize sampling** t+1——状态已最新，构建 mask 并采样。

采样 t+1 排在 commit t 之后，因为 commit 才让 t+1 的 mask 正确——称为 **commit-before-finalize** 顺序。GPU 在第 2、3 相期间持续跑 t+1 的 forward，commit 从关键路径上消失。

普通文本无 mask，forward 和采样都可提前一步；受约束序列 forward 仍然提前，采样等待上一次 commit——同一套循环处理两种情况，无需特判。

## 5. 机制三：Zombies——早收尾、晚释放

launch t+1 前要先定 batch 成员，而这发生在 commit t 之前。若某序列在 t 步命中停止符、却已被烘进 t+1 的 forward 怎么办？GPU 工作无法撤回——序列已结束，但物理上仍在执行中的 batch 里。

Photon 称之为 **zombie**，不加取消逻辑，而是用两个 per-sequence 字段让行为自然涌现：

- `finalized`：命中 EOS 或长度上限后置 `True`；
- `inflight_refs`：仍引用该序列的在途步数（0、1 或 2）。

![已结束序列作为 zombie 搭车 t+1](../_assets/Inference-%E6%8E%A8%E7%90%86%E5%9F%BA%E7%A1%80%E8%AE%BE%E6%96%BD/photon-zombie-lifecycle.svg)

步骤 t commit 检测到 EOS 时：标记 `finalized`、发出结果，但**不拆除**——`inflight_refs` 仍非零（t+1 还引用它）。t+1 的 commit 发现其已 `finalized`，**直接跳过**：不追加 token、不改状态。zombie 无害搭车——占一行、写点没人读的 KV。直到 `inflight_refs` 归零才释放其 KV pages 和 LoRA slot。

这套 finalize-early、release-late 的少量引用计数，替代了「中途取消 batch 里某一行」的丛生的特判逻辑。

## 6. Prefill 复用同一条流水线

真实服务循环同时做两种工作：**prefill**（处理新请求的 prompt+图像，多 token 的一次性重 forward）与 **decode**（为所有进行中的请求逐 token 生成）。

Photon 不把两者分开：prefill 只是同一条双 slot 流水线中 `kind="prefill"` 的 launch。流水线只关心 slot 是否空闲、不关心上个任务类型——prefill forward 可以灌进一个 slot，同时另一 slot 的 decode 正在 commit，反之亦然。同样的 commit 顺序与 `inflight_refs` 簿记跨两种工作保持正确性，zombie 与受约束解码逻辑都无需为「有 prefill 在飞」特判。

**这在短输出场景最重要**：只吐 3 个 token 的请求，生命期几乎全花在 prefill 与准入上；大量短请求的负载本质上是「prefill 流中撒一点 decode」。共享一条流水线让这股流能重叠自己的 CPU 簿记，而不是 prefill↔decode 来回串行。

## 7. 气泡的成本模型与实测

一个 decode 步骤由三部分组成：

- **forward**：重 GPU 矩阵乘。decode 时是**显存带宽瓶颈**——每个 token 都要把全部权重流过计算核心，下限约 `weight_bytes / memory_bandwidth`，显存越快或模型越小它越短；
- **sampling**：分数变成确定 token——mask、argmax/sample、spatial decode、结果 D2H 拷贝，全是 GPU 工作；
- **bookkeeping**：外围 CPU 工作——选下一批（plan）、launch graph、commit 上一步。

阻塞循环三者串行，GPU 在 bookkeeping 期间闲置（即气泡）；流水线把一步的 bookkeeping 滑到下一步的 forward+sampling 之下，周期收敛到 `forward + sampling`。稳态中位数实测（moondream2，单位 ms）确认 GPU 几乎全程忙碌（残余空闲 < 0.05ms）：

| | forward (ms) | sampling (ms) | period (ms) |
| --- | --- | --- | --- |
| 3090 · 1 stream | 4.87 | 0.20 | 5.10 |
| 3090 · 8 streams | 6.66 | 0.27 | 6.97 |
| 3090 · 32 streams | 10.24 | 0.26 | 10.52 |
| B200 · 1 stream | 2.45 | 0.14 | 2.63 |
| B200 · 8 streams | 3.12 | 0.14 | 3.30 |
| B200 · 32 streams | 3.80 | 0.14 | 3.98 |

收益由两股力量拉锯决定——藏起来的气泡 vs 提前跑的代价：

$$\mathrm{speedup} = \frac{T_{\mathrm{block}}}{T_{\mathrm{pipe}}} \times (1 - z)$$

- 第一项是收益：同一步阻塞耗时与流水线耗时之比，即簿记被压到下面后步骤快了多少；
- 第二项 z 是**zombie 税**（机制三的代价）：提前 launch t+1，刚结束的序列还有一个在飞 forward，浪费一步。单流时约每生成 L 个 token 浪费一次 forward，L ≈ 110 时约 1%；但 batch 起来后几乎消失——zombie 只是已按全价流过权重的一步中多出来的一行，搭车近乎免费。**单流处税最重，而吞吐所在之处恰好消退**。

![B200 上实测的阻塞 vs 流水线逐步时间线](../_assets/Inference-%E6%8E%A8%E7%90%86%E5%9F%BA%E7%A1%80%E8%AE%BE%E6%96%BD/photon-decode-timeline.svg)

模型预测与实测对比（depth-1 阻塞 vs depth-2 流水线，其余不变）：

| | blocking (ms) | pipelined (ms) | L | 预测 | 实测 |
| --- | --- | --- | --- | --- | --- |
| 3090 · 1 stream | 5.44 | 5.10 | 104 | +5.7% | +6.5% |
| 3090 · 8 streams | 7.52 | 6.97 | 113 | +7.6% | +7.8% |
| 3090 · 32 streams | 11.74 | 10.52 | 113 | +11.1% | +11.6% |
| B200 · 1 stream | 3.11 | 2.63 | 115 | +17.2% | +17.6% |
| B200 · 8 streams | 4.04 | 3.30 | 115 | +22.2% | +21.9% |
| B200 · 32 streams | 5.55 | 3.98 | 104 | +39.1% | +35.4% |

三个读数要点：

1. **收益随 GPU 变快而增长**：同负载 32 流下 3090 为 +12%，B200 为 +35%。簿记开销与 GPU 速度无关，forward 越短（显存越快/模型越小）气泡占比越大。流水线化是「GPU 变快的保险」——对 Moondream 而言等价于「模型变小」。
2. **Zombie 税真实但小且可摊薄**：单流约 1%（L≈110）；32 流时 3090 的实测 +11.6% 恰好落在无 zombie 的每步比值上——税在单流处咬人，恰在吞吐所在处消退。（B200 32 流低于预测数点的原因更平凡：每步约 4ms，整个运行不足半秒，prefill 与收尾的 batch 收缩占了可见的一段墙钟时间。）
3. **气泡确实可藏才有收益**（曾借此抓到一个 bug：流水线数字曾掉到阻塞水平，追查发现构建受约束解码 mask 时误用同步拷贝；移到 copy stream 后 3090 +11%、B200 +34%）。

## 8. 要点总结

整套技术就三件事：ping-pong slots 防两步碰撞、forward/采样拆分让受约束解码也能提前跑、少量 zombie 引用计数让结束的请求干净拆除。GPU 不再等 CPU，换来百分之几到三分之一的提升——加速器/模型越快，提升越大。

但 Photon 快不是因为这一项技术，而是服务栈上几十处细节的复合：图像缩放切分、模型 kernel、调度顺序、热路径上移除的同步点……没有哪一处是全部答案，足够多处对齐时栈才会快。

## 9. 相关页面

- [Hy3 preview Hopper 推理优化实践](%E6%8E%A8%E7%90%86%E6%80%A7%E8%83%BD%E5%85%A8%E6%A0%88%E4%BC%98%E5%8C%96%E5%AE%9E%E8%B7%B5%EF%BC%88Hopper%20%E6%A1%88%E4%BE%8B%EF%BC%89.md) — 其「MTP 与异步调度优化」是同一思想在投机解码（动态接收长度）下的推广：按最大长度提前准备、用真实结果事后修正
- [推理全流程串讲（概览篇）](LLM%20%E6%8E%A8%E7%90%86%E5%85%A8%E6%B5%81%E7%A8%8B%E6%A6%82%E8%A7%88.md) — prefill/decode 分离与请求生命周期背景
- [KV Cache 内存管理（PagedAttention）](KV%20Cache%20%E5%86%85%E5%AD%98%E7%AE%A1%E7%90%86%EF%BC%88PagedAttention%EF%BC%89.md) — zombie 释放的 KV pages 即 PagedAttention 的分页管理单元
