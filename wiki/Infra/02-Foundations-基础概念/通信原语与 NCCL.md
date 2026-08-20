> 信源：锦恢《AI Infra 软核教程（三）通信原语与 NCCL》，[知乎原文](https://zhuanlan.zhihu.com/p/2070850476703273991)（2026-08-12）。本页为该文的网状知识提炼，非全文转载。

通信原语是对 [GPU 集群硬件拓扑](GPU%20%E9%9B%86%E7%BE%A4%E5%9F%BA%E6%9C%AC%E7%BB%93%E6%9E%84%E4%B8%8E%E5%A4%9A%E6%9C%BA%E7%BB%84%E7%BD%91%E6%8B%93%E6%89%91.md) 的**第一层软件抽象**：忘掉 PCIe、NVLink、IB 的细节，只保留"GPU 之间通信速率不同"这一个事实。抽象层级为 `硬件拓扑 → 通信原语/NCCL → 分布式算法（TP/PP/DP）`，本文覆盖中间层：原语协议本身、前端 PyTorch、后端 NCCL 及其通信算法的复杂度分析。

---

## 一、分布式进程标识

管理和识别集群中所有 GPU 的通用术语（以 2 台 8 卡服务器为例）：

| 术语 | 含义 | 示例值 |
| --- | --- | --- |
| `world_size` | 总进程数（一进程对应一 GPU，故等于总 GPU 数） | 16 |
| `rank` | 进程/GPU 在全局的编号 | rank0 ~ rank15 |
| `local_rank` | 进程/GPU 在**所在机器**上的编号 | 每台机器 0~7 |
| `local_world_size` | 每台机器的进程数 | 8 |
| `node_rank` | 机器编号 | 0、1 |
| `nnodes` | 机器总数 | 2 |

![](../_assets/Foundations-%E5%9F%BA%E7%A1%80%E6%A6%82%E5%BF%B5/distributed-rank-identifiers.webp)

编程时这些值以环境变量形式注入每个进程（由 infra 平台在创建容器时注入）：

```python
world_size = os.getenv('WORLD_SIZE')
rank = os.getenv('RANK')
local_rank = os.getenv('LOCAL_RANK')
node_rank = os.getenv('NODE_RANK')
nnodes = os.getenv('NNODES')
```

**调试应用**：分布式最高频的 bug 是 hang（程序卡死）。常用技巧是在日志打印中带上 `[{local_rank}/{rank}]` 标头；若日志停在 `[2/16]`，即可定位是第 0 台机器的第 2 个进程出了问题。

## 二、为什么需要通信原语

操作系统教科书的进程间通信（IPC：信号量、消息队列、内存映射、管道）不能直接用于 AI Infra 场景：

1. 这些机制操作的是**系统内存**而非 GPU 显存；
2. 未考虑**多机多进程**情形；
3. 并非为分布式计算设计，通信模型描述复杂，不利于调度与规划。

因此业界迭代出一套泛用的**通信原语**：如同编程语言的 `if/for` 一样基础，却能组合表达全部并行通信需求。原语是**与硬件无关的抽象**——纯 CPU 超算、GPU 集群、国产 NPU 顶层都是近乎同一套接口。原语分两类：**P2P 通信**与**集合通信**。

![](../_assets/Foundations-%E5%9F%BA%E7%A1%80%E6%A6%82%E5%BF%B5/p2p-vs-collective.webp)

## 三、P2P 通信

点对点通信：数据从一个 rank 流向另一个 rank。共四种基本操作：

| 名称 | 说明 |
| --- | --- |
| Send | 阻塞发送。向指定 rank 发送 tensor，完成后才继续执行 |
| Recv | 阻塞接收。从指定 rank 接收 tensor，完成后才继续执行 |
| iSend | 非阻塞发送。立即返回可 `wait()` 的 Work 对象，可先执行其他任务 |
| iRecv | 非阻塞接收。立即返回 Work 对象，接收完成前不要读缓冲区 |

要点：

- Send/Recv 必须**成对出现**在两个进程上：`rank0: Send(1)` ↔ `rank1: Recv(0)`。
- 阻塞语义意味着：若 rank0 先到 Send 而 rank1 尚未到 Recv，rank0 就 hang 在 Send 上；若 rank1 在到达 Recv 前崩溃或跳过了 Recv，rank0 **永远卡住**——这是 hang 的典型成因。
- iSend/iRecv 的异步语义在通信调度优化（如计算-通信 overlap）中很有用。

## 四、集合通信

N 张卡按规律同时执行对称的集体操作，用 P2P 描述既繁琐又负载不均。

**引例——数据并行（DP）的梯度同步**：4 张卡各存一份模型副本，batch 32 平摊为每卡 8 条。反向传播后各卡梯度不同，必须汇总求平均再更新，否则各卡训练的就不是同一个模型。

![](../_assets/Foundations-%E5%9F%BA%E7%A1%80%E6%A6%82%E5%BF%B5/data-parallel-dp.webp)

用 P2P 实现（全部发给 rank0 → 算均值 → 发回）的问题：rank0 承担全部通信与计算，其余卡空闲；同步串行传输；且逻辑上对称的操作被写成了不对称的代码。集合通信正是为这类"对称"需求设计的抽象。

> 集合通信只从数学层面规定"一次通信达成什么效果"（输入/输出），**不指定通信拓扑与实现**——后者是通信后端的事。

![](../_assets/Foundations-%E5%9F%BA%E7%A1%80%E6%A6%82%E5%BF%B5/collective-ops-overview.webp)

### Broadcast

从一个 rank 把数据广播给所有 rank，纯通信无计算：`rank0: A → 全部 rank: A`。

### Reduce 与 AllReduce

**归约（Reduce）**的数学本质：给定一组数据和满足**结合律**的二元操作符 $\oplus$，计算

$$x_{0} \oplus x_{1} \oplus \dots \oplus x_{n-1}$$

结合律即：

$$\forall a, b, c \in S \quad (a \oplus b) \oplus c = a \oplus (b \oplus c)$$

（群论中的半群结构。加法→求和、乘法→求积、max→求最大值都是归约。）

- **Reduce**：各 rank 数据归约后结果只发给指定 rank：`A, B, C, D → rank0: A+B+C+D`。
- **AllReduce**：归约结果**每个 rank 都保留一份**。DP 梯度同步因此只需一行：`AllReduce(x, ReduceOp.SUM)`——优雅、对称、简洁、高效。
- AllReduce 有 **barrier 语义**：组内所有 rank 都等待归约与分发完成才继续；一个 rank 失败则全部卡住。NCCL 后端的 barrier 正是用 AllReduce 实现的。

### Scatter 与 Gather / AllGather

- **Scatter**：源 rank 把数组分片分发，每个 rank 得一片：`[A,B,C,D] → rank0: A, rank1: B, …`。处理对象是 tensor 数组。
- **Gather**：Scatter 的互逆操作，各 rank 分片收集到目标 rank：`A, B, C, D → rank0: [A,B,C,D]`。
- **AllGather**：收集结果所有 rank 各一份。

**AllReduce vs AllGather**：从协议结果看 AllGather 再加法也能实现梯度同步，但二者只是"协议相似"，硬件上的通信拓扑规划差别很大——实际总是选 AllReduce，原因就在后端的实现（见 Ring 算法一节）。

## 五、通信前端：PyTorch

现代 HPC 软件很少直接操作硬件 SDK，而是套一层与硬件无关的**通信前端**；厂商的驱动层实现则为**后端**。

| 前端 / API | 典型写法 | 特点 |
| --- | --- | --- |
| MPI | `MPI_Allreduce`、`MPI_Send` | HPC 最经典标准的通信 API |
| PyTorch Distributed | `dist.all_reduce()`、`dist.all_gather()` | AI 训练最常用 |
| Horovod | `hvd.allreduce()` | 面向分布式深度学习，接近 MPI |
| TensorFlow Distributed | `tf.distribute` / `ReplicaContext.all_reduce()` | TF 分布式抽象 |
| JAX / XLA | `lax.psum`、`lax.all_gather` | 强调 SPMD / 编译器式并行 |
| NCCL C API | `ncclAllReduce()` | 已接近底层，不算严格前端 |

Megatron-LM、DeepSpeed 等主流训练框架均选 PyTorch 作通信前端。

### 启动：torchrun 与 rendezvous

多机训练需要各节点先在约定"集合点"握手——rendezvous 节点（rdzv，一般基于 TCP）。`torchrun` 是 PyTorch 官方分布式启动器，自动完成注入环境变量、连接 rdzv 等胶水工作：

```bash
torchrun \
  --nnodes=2 \
  --nproc-per-node=8 \
  --rdzv-id=my_first_c10d \
  --rdzv-endpoint=10.0.0.1:29500 \
  train.py
```

两台服务器各执行一次即可。`--rdzv-id` 是任务编号，`--nproc-per-node` 是每机 rank 数。

### 集群管理平台：Slurm / K8S

真实业界用 Slurm 或 K8S 类平台：提交任务时指定卡数与配置，把 torchrun 命令作为每节点启动代码，平台自动排队、分卡、建容器、聚合日志。

- **优点**：无需关心底层网络细节即可跑多卡任务。
- **代价**：debug 效率显著降低——每次改代码都要重新提交任务（告别 Ctrl+C + 回车的秒级重启）；平台聚合的日志本地 AI 工具看不到（除非平台提供 CLI/MCP），排错退回"半古法编程"。

### 最小代码骨架

```python
import torch.distributed as dist

dist.init_process_group(backend="nccl")  # 依据 torchrun 注入的环境变量建通信组
x = torch.tensor([1], device="cuda")
dist.all_reduce(x)  # 默认加法归约
```

- `dist.get_rank()` / `dist.get_world_size()` 查询当前进程标识。
- **后端与设备绑定**：NCCL 后端只能传 GPU tensor；CPU tensor 通信需换 Gloo 后端。

| Backend | 底层实现 | GPU | CPU | 典型场景 |
| --- | --- | --- | --- | --- |
| NCCL | NVIDIA Collective Communications Library | ✅ | ❌ | 大模型训练主流 |
| Gloo | Meta 开源通信库 | ✅（有限） | ✅ | CPU 训练、调试、小规模 GPU |
| MPI | Message Passing Interface | 取决于实现 | ✅ | 超算/HPC 环境 |

### 常用集合通信写法

```python
dist.broadcast(tensor, src=0)                      # 广播
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)      # AllReduce 求和
dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM)   # Reduce 到 rank0
tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
dist.all_gather(tensor_list, tensor)               # AllGather
```

### 排障技巧：最小通信案例

训练脚本 hang 时不一定是代码问题，可能是环境（硬件/驱动/版本）。用最小案例快速二分：

```python
dist.init_process_group("nccl")
tensor = torch.tensor([dist.get_rank() + 1], device="cuda")
print("enter")
dist.all_reduce(tensor)
print("finish", dist.get_rank(), tensor)
```

卡在 `enter` → 环境问题；通过 → 训练代码问题。

## 六、通信后端：NCCL 基础

**NCCL（NVIDIA Collective Communications Library）** 是 NVIDIA 为 GPU 集群设计的开源 C++ 通信库（[NVIDIA/nccl](https://github.com/NVIDIA/nccl)），事实行业标准（对应华为 HCCL）。

### 初始化与算法选择

执行 `dist.init_process_group("nccl")` 时，NCCL 做 **topology discovery（拓扑发现）**：确认全部 GPU 的连接关系——哪些有 NVLink、哪些同 NUMA、哪些同机器（相当于多机版的 `nvidia-smi topo -m`）。

执行 `dist.all_reduce(tensor)` 时，NCCL 并非立刻通信，而是调用 `ncclAllReduce` 登记操作压入**调度队列**；调度时根据三要素计算通信算法：

1. 初始化获得的拓扑信息（任意两卡连接方式与带宽）；
2. 通信原语类型（如 AllReduce）；
3. 单 rank 通信数据量（如 1 GB）。

相同规格的算法选择结果会被**缓存**，不会每次重算。也可注入环境变量跳过自动选择（调试链路、探明拓扑时有用）：

```bash
export NCCL_ALGO=Ring
```

### α-β 模型（Hockney 模型）

通信分析的标准模型。定义： $p$ = rank 数， $n$ = 单 rank 单次传输数据量， $\alpha$ = 发起一次通信的固定延迟， $\beta$ = 传输 1 Byte 所需时间。一次通信开销：

$$T = S(p) \cdot \alpha + F(p) \cdot n \cdot \beta$$

其中 $S(p)$ 描述 **Latency**， $F(p) \cdot n$ 描述 **Bandwidth**。分析目标就是搞清每个算法的 Latency 与 Bandwidth 关于 $p, n$ 的复杂度。下文统一用 DP 梯度同步为例：记 $g_{ij}$ 为第 $i$ 个 rank 上第 $j$ 层网络参数的梯度（共 4 层）。

## 七、NCCL 通信算法

| 算法 | 说明 | 适用原语 | 典型场景 |
| --- | --- | --- | --- |
| Ring | 环形通信 | AllReduce、AllGather、ReduceScatter、Broadcast、Reduce 等 | 大消息、高带宽，最常用 |
| Tree | 双二叉树 | 主要 AllReduce | 小消息、低延迟 |
| CollNet | 分层集合通信 | 主要 AllReduce，部分 AllGather/ReduceScatter 变体 | 大规模多节点 |
| NVLS | NVLink SHARP / NVLink Switch 加速 | AllReduce、AllGather、ReduceScatter 等 | NVSwitch/NVLink 高端系统 |
| PAT | 分块流水化树形算法 | AllGather、ReduceScatter | 大规模 rank 下小/中消息 |

### 基准：暴力中心化通信

先 gather 全部数据到 rank0，算完均值再 broadcast 回去。rank0 下行带宽是瓶颈， $p-1$ 份数据经它接收：

$$T_{\text{gather}} = \alpha + (p - 1) \cdot n \cdot \beta$$

broadcast 与之对称，故总开销：

$$T_{dummy} = 2\alpha + 2(p - 1) \cdot n \cdot \beta$$

$$\mathrm{Latency} = \mathcal{\Theta}(1), \quad \mathrm{Bandwidth} = \mathcal{\Theta}(pn)$$

延迟虽低，但带宽复杂度是**乘积**——规模一大就爆炸。

### Ring

所有卡首尾相连成环，只能沿环方向与后继通信。Ring 实现 AllReduce 分为两步：

1. **ReduceScatter**（ $p-1$ 轮）：每个 rank 拿到某一层梯度的和；
2. **AllGather**（ $p-1$ 轮）：全部层的梯度和扩散到所有 rank。

数据被分为 $p$ 份，每次传 $n/p$；各节点完全对称，均向后继发送 $2(p-1)$ 次：

$$S(p) = 2(p - 1), \quad F(p) \cdot n = 2(p - 1) \frac{n}{p} = 2\left(1 - \frac{1}{p}\right) n$$

$$\mathrm{Latency} = \mathcal{\Theta}(p), \quad \mathrm{Bandwidth} = \mathcal{\Theta}(n)$$

带宽复杂度消掉了乘积，且不受单卡带宽瓶颈影响——大消息场景的最优选择。

### Tree（Double Binary Tree）

把 rank 组织成平衡二叉树：叶子逐层向父节点"交作业"，父节点归约后继续上传，根节点得到全量和再向下分发。

朴素单树的问题：内部节点通信压力大、叶子节点分发阶段闲置，负载不均。NVIDIA 的 **Double Binary Tree** 构造两棵互补二叉树（让内部节点与叶子角色互换），数据分两份各走一棵树：

$$S(p) = 2h = 2\lceil \log_{2} p \rceil, \quad F(p) \cdot n = 2n$$

$$\mathrm{Latency} = \mathcal{\Theta}(\log_{2} p), \quad \mathrm{Bandwidth} = \mathcal{\Theta}(n)$$

保持带宽不变，延迟从 $p$ 降到 $\log_{2} p$——小消息低延迟场景优于 Ring。

### CollNet 与 NVLS：利用硬件差异

前两种算法只考虑逻辑拓扑，而真实硬件中 intra-node 远快于 inter-node。CollNet 与 NVLS 进一步回答：**哪些通信留在高速 intra-node 完成，哪些数据才有必要进入昂贵的 inter-node 网络。**

- **CollNet**：利用网络的 **in-network reduction** 能力（如 NVIDIA SHARP，可在 IB 交换网络中直接完成部分归约）。流程：先在各服务器节点内完成局部归约 → 数据进入网络交换时由 SHARP 完成剩余归约 → 各节点得到最终和。`NCCL_ALGO` 支持 CollNetChain 与 CollNetDirect，区别仅在同机内拓扑组织，Direct 多数场景更好。
- **NVLS（NVLink SHARP）**：Hopper 架构起，NVSwitch 可将部分集合通信 offload 到交换芯片中执行。

> 真实执行中各阶段并非严格串行：NCCL 把数据切成 chunk，让 intra-node 传输、NIC 发送、网络归约、结果返回组成 overlap 流水线。

## 八、小结

- 原语分层：通信原语（协议）→ 前端（PyTorch `torch.distributed`）→ 后端（NCCL）→ 硬件拓扑。
- 算法选型直觉：**大消息用 Ring（带宽优），小消息用 Tree（延迟优），有 SHARP/NVSwitch 硬件则交给 CollNet/NVLS**。
- 下一章将进入分布式训练算法（DP/TP/PP 等），它们是建立在 NCCL 之上的又一层抽象。

## 相关笔记

- **硬件基础**：[GPU 集群基本结构与多机组网拓扑](GPU%20%E9%9B%86%E7%BE%A4%E5%9F%BA%E6%9C%AC%E7%BB%93%E6%9E%84%E4%B8%8E%E5%A4%9A%E6%9C%BA%E7%BB%84%E7%BD%91%E6%8B%93%E6%89%91.md) — 系列第二篇：本文"拓扑发现"与 intra/inter-node 差异的硬件来源。
- **领域全景**：[AI Infra 领域概览](AI%20Infra%20%E9%A2%86%E5%9F%9F%E6%A6%82%E8%A7%88.md) — 系列第一篇：通信原语在五大板块中的位置。
- **张量并行**：[张量并行（模型并行）](../03-Training-%E8%AE%AD%E7%BB%83%E5%9F%BA%E7%A1%80%E8%AE%BE%E6%96%BD/%E5%BC%A0%E9%87%8F%E5%B9%B6%E8%A1%8C%EF%BC%88%E6%A8%A1%E5%9E%8B%E5%B9%B6%E8%A1%8C%EF%BC%89.md) — TP 的通信正是 AllReduce，可对照本文 Ring/Tree 复杂度理解其开销。
- **训练框架**：[Megatron-LM 论文精读](../03-Training-%E8%AE%AD%E7%BB%83%E5%9F%BA%E7%A1%80%E8%AE%BE%E6%96%BD/Megatron-LM%20%E8%AE%BA%E6%96%87%E7%B2%BE%E8%AF%BB.md) — f/g 通信算子即 AllReduce 在层内模型并行中的具体形态。
