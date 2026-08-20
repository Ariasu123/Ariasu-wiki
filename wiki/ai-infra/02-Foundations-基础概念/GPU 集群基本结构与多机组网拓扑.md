> 信源：锦恢《AI Infra 软核教程（二）GPU 集群基本结构与多机组网拓扑》，[知乎原文](https://zhuanlan.zhihu.com/p/2066185773435589967)（2026-08-02）。本页为该文的网状知识提炼，非全文转载。

大模型训练集群"好不好用"，一半取决于**通信性能**。理解并行算法（TP/PP/DP）和 NCCL 之前，必须先理解底层硬件拓扑：单机内 GPU 如何互联（PCIe → NVLink → NVSwitch），多机之间如何组网（IB 卡 → Rail → 多层 Clos）。本文默认讨论上下文为 **N 台 8 卡 GPU 服务器**；带宽除特别说明外均指**单向带宽**。

系列前篇见 [AI Infra 领域概览](AI%20Infra%20%E9%A2%86%E5%9F%9F%E6%A6%82%E8%A7%88.md)。

---

## 一、GPU 服务器硬件基础

一台大模型训练服务器的主要组成：**CPU、系统内存、GPU、PCIe 总线、网卡、本地存储**，逻辑结构与普通 PC 相同。

![](../_assets/Foundations-%E5%9F%BA%E7%A1%80%E6%A6%82%E5%BF%B5/gpu-server-structure.webp)
### 为什么主流是单机 8 卡

- **供电**：满载 8 卡 B300 功率约 14.5 kW（≈10 台吹风机同时工作），已接近单机供电极限。
- **散热**：大功率风冷噪音极大（进机房需隔音耳机）；液冷（冷却液浸泡主板）安静但更贵。
- **并行算法友好**：卡数最好是 2 的次幂（8、16），便于并行切分。
- **故障率**：32 卡服务器一张卡出问题需全机停转排查，故障率与损失远大于 8 卡（类似火箭推进器数量引入的系统复杂度）。
- **通信路径不均匀**：卡越多，卡间通信拓扑越难做到对称（见下文）。

单机 16 卡在部分高端云厂商库存中可见，是未来趋势；本系列后续算法讨论均以 8 卡服务器为单位。

## 二、单机内通信：PCIe 与 PCIe Switch

### NUMA 架构

单机八卡服务器通常是**双路**（两个 CPU 插槽）：每个 CPU 连同同侧 4 张卡、内存等构成一个 **NUMA（Non-Uniform Memory Access，非一致内存访问）节点**。

- NUMA 节点内：CPU 与 PCIe 设备（GPU、内存）相互访问一致。
- 跨 NUMA：需走 CPU 厂商专用的 CPU2CPU 协议——Intel 为 **UPI**（Xeon 6 的 UPI 2.0 单链路最高 24 GT/s），AMD 为 **Infinity Fabric**。跨 NUMA 通信**延迟更高、有效带宽更低**。

![NUMA 双路八卡服务器架构](../_assets/Foundations-%E5%9F%BA%E7%A1%80%E6%A6%82%E5%BF%B5/numa-dual-socket-8gpu.png)

### PCIe 直连的瓶颈

TP（张量并行）等并行算法会在相邻 GPU 间频繁通信。无优化时通信路径为：

```text
GPU 0 ── CPU Root Complex ── GPU 1
```

由 DMA 驱动完成；若系统不支持 **PCIe P2P**，还需借 **pinned memory** 中转，通信数据量大时内存成为主要瓶颈。

带宽数据（PCIe 5.0 x16）：

- 理论单向带宽 ≈ **63 GB/s**；
- 实际 GPU-to-GPU 有效带宽受接口代际、PCIe 拓扑、Root Complex、P2P 支持、链路竞争影响，实测可能只有 **23–29 GB/s**。

### PCIe Switch

在同 NUMA 下的所有 GPU 之间加入 **PCIe Switch**，让 GPU 绕开 pinned memory 直接在 Switch 内横向通信：

```text
GPU 0 ── PCIe Switch ── GPU 1
```

带宽提升至 **48–58 GB/s**（接近翻倍），但仍不足以喂饱训练——在作者的简化估算（48 层 decoder-only、TP=8、hidden size 8192、序列长 4096、BF16、每层 4 次 ring all-reduce）下，通信时间占比仍约 42%–65%。且 Switch 到 CPU Root 的上行链路总带宽有限，多卡满载时会产生竞争。

![PCIe P2P 路径与 PCIe Switch 拓扑对比](../_assets/Foundations-%E5%9F%BA%E7%A1%80%E6%A6%82%E5%BF%B5/pcie-p2p-switch-topology.png)

### 查询本机拓扑：`nvidia-smi topo -m`

```text
GPU0  GPU1  GPU2  GPU3  GPU4  GPU5  GPU6  GPU7  CPU Affinity
GPU0     X    PIX   PXB   PXB   SYS   SYS   SYS   SYS   0-31
GPU1    PIX    X    PXB   PXB   SYS   SYS   SYS   SYS   0-31
...
```

| 标记 | 含义 |
| --- | --- |
| `PIX` | 两张 GPU 之间最多经过一颗 PCIe Switch |
| `PXB` | 经过多级 PCIe Bridge/Switch，但不经过 CPU Host Bridge |
| `PHB` | 路径经过 PCIe Host Bridge / CPU Root Complex |
| `NODE` | 经过同一 NUMA 节点内的多个 Host Bridge |
| `SYS` | 跨 NUMA 节点或跨 CPU Socket |
| `NV#` | 通过若干条 NVLink 连接 |
| `X` | 当前 GPU 自身 |

NCCL 兼容性测试、排查通信性能问题时，第一步通常就是看这张表。

## 三、NVLink 与 NVLink Bridge

PCIe 5.0 x16 的理论极限就是 63 GB/s，不换代就无法突破。NVIDIA 于 2016 年在 P100 上推出 **NVLink** 专用高带宽互联：P100 最多 4 条 NVLink 1.0，每条双向聚合 40 GB/s，整卡合计 160 GB/s。

**A100 PCIe 版**：卡上有 3 个 NVLink 接口（塞盖下），用 **NVLink Bridge** 连接相邻两块 A100：

- 单个 Bridge 双向带宽 **200 GB/s**（单向 100 GB/s），三个合计 600 GB/s；
- 每张 A100 只能与**唯一一个邻居**两两互联，形成"孤岛"对。

为什么只允许两两连接？若 GPU0—GPU1、GPU1—GPU2 各用一桥，GPU0 到 GPU2 需经 GPU1 中转，通信延迟不一致、拓扑不对称，Bridge 插槽顺序难分配，NCCL 算法也难以施行。对称拓扑是集合通信高效的前提。

8 张 A100 PCIe 两两互联后的三类通信路径：

![8×A100 PCIe + NVLink Bridge 两两互联拓扑](../_assets/Foundations-%E5%9F%BA%E7%A1%80%E6%A6%82%E5%BF%B5/a100-pcie-nvlink-bridge-pairs.png)

![](../_assets/Foundations-%E5%9F%BA%E7%A1%80%E6%A6%82%E5%BF%B5/a100-pcie-nvlink-topology.webp)

```text
GPU0 --> NVLink --> GPU1 : 带宽最大，延迟低
GPU0 --> PCIe Switch0 --> GPU1 : 带宽中等，延迟低
GPU0 --> PCIe Switch0 --> CPU0 --> UPI --> CPU1 --> PCIe Switch1 --> GPU1 : 带宽中等，延迟高
```

此时 `nvidia-smi topo -m` 中 NVLink 对显示为 `NV3`（3 条 NVLink），跨 NUMA 仍为 `SYS`。

**工程启示**：TP 单次通信量大但频率低，PP 单次通信量小但频率高——利用通信特征在不对称拓扑上合理规划通信路径，可最大化训练效能；NCCL 会自动完成大部分路径选择，极端场景才需要硬编码通信方案。

## 四、SXM 封装与 NVSwitch

### A100 的两种物理封装

| 封装 | 形态 | 互联能力 | 成本 |
| --- | --- | --- | --- |
| A100 PCIe | 标准金手指卡 | NVLink Bridge 两两互联 | 基准 |
| A100 SXM | 专用模组（无金手指） | 配合 NVSwitch 全互联 | 价格近 PCIe 版两倍，且必须配 HGX/DGX 底板 |

**底板（Baseboard）**：专门集成 SXM GPU、NVLink 走线和 NVSwitch 的 GPU 底板，本身不是完整服务器主板。NVIDIA 提供底板设计标准，下游服务器厂商再将其整合进整机。H200、B200、B300 等后续计算卡均有 SXM 形态。

### NVSwitch：单机八卡全互联的终极方案

NVSwitch 是专门转发 NVLink 数据的**交换芯片**。DGX A100 用 **6 颗 NVSwitch** 完成 8 卡互联：

![](../_assets/Foundations-%E5%9F%BA%E7%A1%80%E6%A6%82%E5%BF%B5/dgx-a100-nvswitch.webp)

每张卡从 NVLink 接口引出 12 条线，分摊连接到 6 颗 NVSwitch；任意两卡通信由 NVSwitch"牵线搭桥"，拓扑变为全互联：

```text
GPU0 GPU1 GPU2 GPU3 GPU4 GPU5 GPU6 GPU7
GPU0     X   NV12 NV12 NV12 NV12 NV12 NV12 NV12
GPU1    NV12  X   NV12 NV12 NV12 NV12 NV12 NV12
...
```

SXM 形态下 NVLink 不再是独立的 Bridge 硬件，而是"模组触点 + 底板内部高速 PCB 铜走线"——这也是 8 卡高速互联必须用特制底板、无法在主板上打补丁的原因。

## 五、多机通信与 InfiniBand

- **intra-node**：单机多卡内部通信（上文全部内容）；
- **inter-node**：多机之间的通信，是本节主题。

### 为什么普通以太网不够用

1. **带宽**：传统服务器 100 Gb/s 网口 ≈ 12.5 GB/s；现代 AI 服务器的 200/400/800 Gb/s 网络 ≈ 25/50/100 GB/s，仍远低于 NVLink。
2. **中转开销**：网卡设计之初不支持直接读写显存，数据须 GPU → 内存 → 内核网络协议栈 → 对端，带来四层问题：CPU 内存复制消耗带宽、内核协议栈增加延迟、CPU 持续参与搬运、大规模 collective 易受网络拥塞影响。

### InfiniBand 简史

- 上世纪 90 年代，CPU 变强但服务器间/服务器与存储间数据通道成为瓶颈，出现两套竞争方案：Intel、Microsoft、Sun 推动的 **NGIO**，与 IBM、Compaq、HP 推动的 **Future I/O**。
- 1999 年，为避免资源浪费和市场分裂，几家公司成立 **IBTA（InfiniBand Trade Association）**，制定 **InfiniBand** 行业标准，覆盖网卡、线缆到软件栈。
- **Mellanox** 成为 IB 最重要的商业实现厂商，其 **HCA 网卡**支持 **RDMA**：两台服务器可绕过 CPU、在授权下直接交换内存数据。
- 2019 年 NVIDIA 以约 69 亿美元收购 Mellanox（2020 年以 70 亿美元完成），将 IB 技术纳入其 AI 生态壁垒。NVIDIA 真正的护城河不是单卡算力，而是 CUDA + NCCL + 网络软硬件的超级生态。

### ConnectX 系列 IB 卡

- DGX A100 官方配置为 8 张 **ConnectX-6 VPI** 单端口适配器（VPI 卡也可切换为普通以太网卡模式）。
- 每端口提供 **25 GB/s** 跨机带宽；单端口版独享 PCIe x16，双端口版高峰期可能在 PCIe x16 上发生竞争。
- 每台服务器 8 张 IB 卡 ↔ 8 张 GPU，两机之间形成 8 条高速互联通道，多机多卡即可抽象为"服务器 ↔ 交换机"的组网问题。

## 六、Clos 网络拓扑：从几百卡到万卡

### 单层 Clos：8-Rail

单台交换机端口有限：以配套 ConnectX-6 的 **QM8700** 为例，仅 40 个 25 GB/s 端口，而一台 DGX A100 要占 8 口——一台交换机最多接 5 台服务器。

解法是把网络拆成多个**平面**：每台服务器的第 N 张 IB 卡接入第 N 台交换机，所有服务器的第 N 张卡 + 第 N 台交换机构成一个独立的 **Rail**（8 卡机 → 8 个 Rail）。

![](../_assets/Foundations-%E5%9F%BA%E7%A1%80%E6%A6%82%E5%BF%B5/8-rail-single-tier-clos.webp)

每个 Rail 内部所有节点彼此可达，任意服务器的任意 GPU 都能互通。按 QM8700 规格，**8-Rail 单层 Clos（One-Tier Clos）最多容纳 40 台服务器 = 320 张 A100**。

### 双层 Clos：Leaf + Spine

更多卡时把 8-Rail 平面视为一个节点递归扩展：与 GPU 服务器直连的交换机叫 **Leaf**，上行交换机叫 **Spine**。

Leaf 交换机的 40 个端口拆为 **32 + 8**：32 口接服务器，8 口预留上行接 Spine。这样"服务器组 + 8 台 Leaf"对外暴露 8×8 个空闲 IB 口，与单台服务器的 8 个 IB 口**递归同构**，可逐层构建：

![](../_assets/Foundations-%E5%9F%BA%E7%A1%80%E6%A6%82%E5%BF%B5/two-tier-clos-leaf-spine.webp)

容量估算：每个 Leaf 组含 32 × 8 = 256 张卡，Spine 层最多接 40 个 Leaf 组 → 40 × 256 = **10240 张卡**（理论万卡集群）。

> 能够组成独立网络的系统称为**网络平面**；此结构下 Rail N = 每个 Leaf 内各服务器的第 N 张卡 + 该 Leaf 第 N 台交换机 + 第 N 台 Spine。

### 三层 Clos：SuperPOD 参考设计

32+8 拆分的隐患：极端情况下 Leaf 下行:上行 = 4:1 失衡，数据进来多送出去慢，有效带宽可下降 75%。NVIDIA 在《DGX SuperPOD 参考架构》中改为 **20 + 20**（上下行带宽一致）；上行端口增多后 Spine 端口不够用，需再加第三层交换机 **Core**。

- **SU（Scalable Unit）**：包含 8 台交换机和 20 台 GPU 服务器的网络平面，是三层 Clos 的基本扩展单元。

NVIDIA 官方拓扑设计建议（A100 时代）：

| DGX 节点数 | A100 数量 | SU 数量 | Leaf | Spine | Core | 网络层级 |
| --- | --- | --- | --- | --- | --- | --- |
| 10 | 80 | 0.5 | 8 | 2 | 0 | 两层 |
| 20 | 160 | 1 | 8 | 4 | 0 | 两层 |
| 40 | 320 | 2 | 16 | 10 | 0 | 两层 |
| 80 | 640 | 4 | 32 | 20 | 0 | 两层 |
| 100 | 800 | 5 | 40 | 20 | 0 | 两层 |
| 120 | 960 | 6 | 48 | 80 | 24 | 三层 |
| 140 | 1120 | 7 | 56 | 80 | 28 | 三层 |

### 万卡及以上：fat tree

三层 Clos 参考方案上限约 1120 卡。更大规模依靠 **fat tree**：通过提供更多冗余交换机，从拓扑上保证任意节点间访问的高效性。满载 fat tree 及其在多层 Clos 上的移植与变种，是目前千卡万卡超节点的主流拓扑方案。

> 注意区分："拥有一万张卡" ≠ "万卡集群"——把一万张卡连成一片高速互联的网络，成本可能接近再买一万张卡。

## 七、一体化解决方案：NVL72

不想自己组网的终极选项：NVIDIA 整机柜集群 **NVL72**（如 Vera Rubin NVL72）：

- 72 张 Rubin GPU，单卡显存 288 GB；
- inter-node 带宽 20 TB/s 以上，intra-node 达 260 TB/s；
- 整机液冷，全功率能耗 ≈ 100 台 1500W 吹风机；
- 2026 年市场成交价约 5800–6800 万元人民币/柜，现货可达 7000 万+/柜。

![](../_assets/Foundations-%E5%9F%BA%E7%A1%80%E6%A6%82%E5%BF%B5/vera-rubin-nvl72.webp)

至此 NVIDIA 计算卡共有三种封装形态：**PCIe**（标准卡）、**SXM**（HGX/DGX 模组）、**NVL**（NVL72 超算专用）。

## 八、软件栈与多机通信方案对比

### OpenFabrics 生态

- **OFA（OpenFabrics Alliance）**：2004 年成立（原名 OpenIB Alliance，专为 IB 做 Linux 适配），围绕 RDMA、内核旁路和高性能网络软件的开放行业组织，后也覆盖以太网 RDMA（如 RoCE）。OpenMPI 等 HPC 项目的上游网络接口多由该生态提供。
- **OFI（OpenFabrics Interface）**：OFA 推行的核心协议；**libfabric** 是 OFI 的实现（AWS EFA 软件栈 aws-ofi-nccl 中的 "ofi" 即指它）。
- **verbs**：让 Linux 用户态程序直接提交 RDMA 操作的底层编程模型，对应库为 libibverbs；如今 verbs 实现主要并入 **rdma-core**，由 Linux RDMA 社区与硬件厂商共同维护。

### 主流多机通信方案

| 方案 | 底层网络 | NCCL 接入路径 | RDMA/GPU Direct | 典型环境 |
| --- | --- | --- | --- | --- |
| InfiniBand | 专用 IB Fabric | NCCL → verbs → IB | 支持 GPUDirect RDMA | NVIDIA DGX、HPC、AI 集群 |
| RoCEv2 | 以太网上的 RDMA | NCCL → verbs → RoCE | 支持 GPUDirect RDMA | 自建以太网 AI 集群 |
| AWS EFA | AWS 专用云 Fabric | NCCL → aws-ofi-nccl → libfabric → EFA | 支持设备 RDMA，因实例而异 | AWS EC2 P 系列 |
| HPE Slingshot | HPE Cray 高性能 Fabric | NCCL → Slingshot NCCL 插件 | 平台相关 | HPE Cray 超算 |
| Google GPUDirect-TCPX/TCPXO | Google 云网络 + gVNIC | NCCL → Google NCCL Net Plugin | GPU 直接数据路径 | Google Cloud A3 |
| Google gIB | Google 云 RDMA Fabric | NCCL/gIB 插件 | 支持高性能 RDMA 路径 | Google A3 Ultra、A4、A4X |

InfiniBand 是目前性能最好、bug 最少的方案——除了贵都是优点。

### 用 NCCL 日志识别当前方案

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH
```

跑训练或 [nccl-tests](https://github.com/nvidia/nccl-tests)，对照日志关键词：

| 多机通信方案 | 典型 NCCL 日志关键词 |
| --- | --- |
| InfiniBand | `NET/IB`、`mlx5_0`、IB、GDRDMA |
| RoCEv2 | `NET/IB`、`mlx5_0`、RoCE、GDRDMA |
| AWS EFA | `NET/OFI`、`NET/Libfabric`、provider efa、GDRDMA |
| 普通以太网 TCP | `NET/Socket`、Using eth0 |

## 九、全文脉络小结

通信带宽的层层升级构成了 GPU 集群的演进主线：

```text
PCIe 直连（23–29 GB/s）
  → PCIe Switch（48–58 GB/s）
    → NVLink Bridge 两两互联（单向 100 GB/s）
      → NVSwitch 单机全互联（SXM 底板）
        → IB 卡 + Rail/Clos 多机组网（单端口 25 GB/s × 8）
          → fat tree / NVL72 万卡级一体化
```

下一篇（系列之三）将在此硬件基础上讲 NCCL 与集合通信原语。

## 相关笔记

- **系列前篇**：[AI Infra 领域概览](AI%20Infra%20%E9%A2%86%E5%9F%9F%E6%A6%82%E8%A7%88.md) — AI Infra 五大板块、显存墙与训推工程难点；本文是硬件层的展开。
- **系列后篇**：[通信原语与 NCCL](%E9%80%9A%E4%BF%A1%E5%8E%9F%E8%AF%AD%E4%B8%8E%20NCCL.md) — 系列第三篇：硬件拓扑之上的软件抽象——P2P/集合通信原语与 NCCL 通信算法。
- **张量并行**：[张量并行（模型并行）](../03-Training-%E8%AE%AD%E7%BB%83%E5%9F%BA%E7%A1%80%E8%AE%BE%E6%96%BD/%E5%BC%A0%E9%87%8F%E5%B9%B6%E8%A1%8C%EF%BC%88%E6%A8%A1%E5%9E%8B%E5%B9%B6%E8%A1%8C%EF%BC%89.md) — 本文 TP 通信案例对应的并行范式：通用切分策略与通信开销分析。
- **训练框架**：[Megatron-LM 论文精读](../03-Training-%E8%AE%AD%E7%BB%83%E5%9F%BA%E7%A1%80%E8%AE%BE%E6%96%BD/Megatron-LM%20%E8%AE%BA%E6%96%87%E7%B2%BE%E8%AF%BB.md) — 层内模型并行的 f/g 通信算子，正是运行在本文所述拓扑之上。
