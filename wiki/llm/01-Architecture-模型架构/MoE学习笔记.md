---
date: 2026-04-20
---
![](../_assets/Architecture-%E6%A8%A1%E5%9E%8B%E6%9E%B6%E6%9E%84/MoE%E5%AD%A6%E4%B9%A0.png)

# MoE (混合专家模型) 学习笔记

关联阅读：可结合 [[10-LLM-大语言模型/01-Architecture-模型架构/MiniMind架构图解析|MiniMind架构图解析]] 理解；MiniMind 的稠密结构可作为理解 MoE 改造的基线。


## 一、 MoE 核心理论与本质概念

### 1. 核心定义与价值

传统稠密模型（Dense Model）在处理任何输入时，都会强制激活全部参数，计算量与参数量完全绑定。**混合专家模型（MoE）巧妙地将模型总容量与激活计算量解耦。** 通过引入路由机制，模型每次只激活极少部分的“专家”网络参与计算，从而在不显著增加推理计算量的前提下，大幅提升模型的参数总容量。

### 2. MoEGate (路由门控) 的核心机制

传统的 FFN 层通常使用 SiLU 等激活函数对特征维度进行门控筛选，而在 MoE 架构中，引入了一套全新的门控筛选机制——**Router（路由器）**，用于对不同的专家进行筛选。

- **计算逻辑：** 对于每个到来的 Token，系统会拿出其前一层的隐藏状态（`hidden_state`）与 gate 的权重进行点积计算。点积越大，表示该 Token 越“偏爱”某个专家。
    
- **Top-K 筛选：** 根据点积得分，模型会为每个 Token 选出得分最高的 `Top_k` 个专家进行权重处理和特征提取。

---

## 二、 负载均衡与双重辅助损失推导

在 MoEGate 的实际运行中，必须防止一个致命问题：某个专家被选择过多，被过度依赖（强者恒强），而其他专家被闲置。计算的核心在于综合考虑“打分”和“频率”，不让某个专家太过被偏爱。针对这一问题，业界提出了两种维度的**辅助损失函数（Auxiliary Loss）**来进行约束：

在深入进行序列级和批级辅助损失推导之前，我们先统一以下核心变量与符号的定义。这些参数构成了 MoE 路由计算的底层逻辑：

**1. 维度与数量参数**

- $B$：**Batch Size**（批次大小），即一次训练迭代中包含的样本数量。
- $L$：**序列长度**（Sequence Length），即每个样本中包含的 Token 数量。
- $N$：**总 Token 数**，计算公式为 $N = B \times L$，代表当前 Batch 中需要处理的 Token 总量。
- $E$：**专家总数**（`n_routed_experts`），即 MoE 层中可供选择的独立前馈神经网络（FFN）的数量。
- $k$：**每个 Token 激活的专家数**（`top_k`），即路由门控网络为每个 Token 选出的得分最高的专家个数（通常 $k \ll E$）。

**2. 路由打分与指示变量**

- $s_{i,e}$：**路由原始分数**。表示第 $i$ 个 Token 对专家 $e$ 的偏好程度（通常是 Router 层的 logits 经过 softmax 激活函数后的概率值）。

- $m_{i,e}$：**路由指示变量**。这是一个离散的布尔/整型变量，$m_{i,e} \in \{0,1\}$。
    - 如果专家 $e$ 被 Token $i$ 的 top-$k$ 机制选中，则 $m_{i,e} = 1$。
    - 如果未被选中，则 $m_{i,e} = 0$。

**3. 核心基准线：理想负载均衡状态**

为了衡量当前的负载是否均衡，我们需要一个“理想基准”。在绝对完美的负载均衡状态下：

- **期望选中次数**：每个专家应该处理同样多的 Token。当前 Batch 总共需要进行 $N \cdot k$ 次专家分配，分摊到 $E$ 个专家头上，每个专家理想情况下应被选中约 $\frac{N \cdot k}{E}$ 次。
- **平均选择率**：这也意味着，任何一个专家被任意一个 Token 选中的平均概率（或频率）应趋近于 $\frac{1}{E}$。

### 1. 序列级辅助损失（Sequence-level Auxiliary Loss, `seq_aux=True`）

该方式以每个序列（句子/样本）为单位计算负载，保证每个句子内部都不会过度依赖某个专家。它非常适用于处理长序列或需要细粒度均衡的场景。

**数学推导与计算步骤：**

1. 统计每个序列中各专家被选中的次数（其中 $b$ 为 batch 索引，$t$ 为序列位置，$j$ 为 top-k 中的第 $j$ 个专家）：
    
    $$c_{b,e} = \sum_{t=1}^{L} \sum_{j=1}^{k} \mathbf{1}(\text{topk\_idx}_{b,t,j} = e)$$
    
2. 归一化为相对负载率（理想均匀负载为 1，若 $\tilde{c}_{b,e} > 1$，则表示专家 $e$ 在该序列中被过度使用）：
    
    $$\tilde{c}_{b,e} = \frac{c_{b,e}}{L \cdot k / E}$$
    
3. 计算该序列中专家 $e$ 的平均打分（$s$ 为 Router 给出的 softmax 概率得分）：
    
    $$\bar{s}_{b,e} = \frac{1}{L} \sum_{t=1}^{L} s_{(b,t),e}$$
    
4. 计算序列级辅助损失（$\alpha$ 为损失权重系数）：
    
    $$\mathcal{L}_{\text{aux}}^{\text{seq}} = \alpha \cdot \frac{1}{B} \sum_{b=1}^{B} \sum_{e=1}^{E} \tilde{c}_{b,e} \cdot \bar{s}_{b,e}$$
    

> **直观解释：** 如果某个专家在某个序列中被高频选中（导致 $\tilde{c}_{b,e} > 1$），且其平均得分 $\bar{s}_{b,e}$ 也高，两者的乘积就会变大，导致整体损失上升。通过梯度反向传播，模型会抑制该专家的打分，促使其被少选；反之，使用不足但分数高的专家会被鼓励多用。

### 2. 批级辅助损失（Batch-level Auxiliary Loss, `seq_aux=False`）

该方式以整个 Batch 为单位统计负载，保证整体不会过度依赖某个专家。这种方式计算更简洁、高效，是 Switch Transformer 等主流方案采用的方式，极其适合大规模训练。

**数学推导与计算步骤：**

1. 计算专家 $e$ 的全局平均选择率（其中 $N$ 为当前 batch 的 token 总数，$m_{i,e}$ 来自展平后的 top-k 索引的 one-hot 编码）：
    
    $$f_e = \frac{1}{N \cdot k} \sum_{i=1}^{N \cdot k} m_{i,e}$$
    
2. 标准化为“相对负载因子”（绝对均衡时 $f_e = 1/E$，乘以 $E$ 后均衡态的 $\hat{f}_e$ 即为 1）：
    
    $$\hat{f}_e = f_e \cdot E$$
    
3. 计算专家 $e$ 的全局平均打分：
    
    $$p_e = \frac{1}{N} \sum_{i=1}^{N} s_{i,e}$$
    
4. 计算批级辅助损失：
    
    $$\mathcal{L}_{\text{aux}}^{\text{batch}} = \alpha \cdot \sum_{e=1}^{E} \hat{f}_e \cdot p_e$$
    

> **直观解释：** 损失项 $\hat{f}_e \cdot p_e$ 构成了直接的惩罚机制。对于高负载的专家（$\hat{f}_e > 1$），系统要求其具有较低的平均得分 $p_e$ 以降低 Loss；而低负载的专家应有较高的分数。门控网络会以此调整打分分布，从而在全局 Batch 层面实现精妙的负载均衡。

---

## 三、 MoE 演进历程

MoE 架构的发展是一部不断追求“规模与效率解耦”的历史：

- **萌芽阶段 (1991年)：** 提出竞争式损失函数，迫使专家网络产生专业化分工。
    
- **RNN与早期Transformer (2017-2021年)：** Google 引入稀疏门控与 Top-k 机制。随后的 Switch Transformer 进一步采用极简的 Top-1 路由，将参数推至 1.6 万亿，确立了“参数量作为独立缩放轴”的理念。
    
- **当前主流趋势：** 由早期如 Mixtral 8x7B 的“大参数、少专家”架构，逐渐演进为如 DeepSeek-V3 的“小参数、多专家”架构，通过更细粒度的专家分工实现极致的推理成本控制与高性能。
    

---

## 四、代码实现

### 1. 初始化阶段

这一阶段主要负责把 MoE 层运行所需的核心组件准备好：

1. 定义门控函数 `nn.Linear`
   - 输入维度是 `hidden_size`。
   - 输出维度是 `num_experts`。
   - 它的作用是为每个 token 生成对各个专家的路由打分。

2. 构建专家集合 `nn.ModuleList`
   - 通常会在 `ModuleList` 中放入多个 `FeedForward`。
   - 每个专家本质上都是一个独立的 FFN。

3. 准备激活函数 `ACT2FN`
   - 用于专家内部的非线性变换。
   - 它决定专家在前向传播时采用哪种激活函数。

---

### 2. 前向传播主流程

这一阶段主要负责把 token 路由到合适的专家，并收集专家输出：

#### 2.1 平铺输入

- 先把输入中的所有 token 平铺成一行。
- 这样做的目的是把 batch 维和序列维合并，统一处理所有 token。

#### 2.2 计算路由得分

- 使用 gate 对每个 token 计算路由分数。
- 这些分数表示当前 token 对不同专家的偏好程度。

#### 2.3 执行 Top-k 路由

- 为每个 token 选出得分最高的前 `k` 个专家。
- 得到：
  - `topk_weight`
  - `topk_idx`
- 这两个张量的形状通常都是 `num_token, k`。
- 其中 `k` 表示一个 token 最终选中的专家数量。

#### 2.4 路由权重归一化

- 对选中的专家权重做归一化。
- 这样后面在聚合专家输出时，权重才有明确的比例意义。

#### 2.5 创建输出容器

- 先创建一个容器，用来存放所有 token 经过专家计算后的结果。
- 后面每个专家处理完自己的 token 后，都会把结果写回这个容器。

#### 2.6 稀疏计算

- 这是 MoE 最关键的一步。
- 核心做法是：
  - 通过 `mask` 这种布尔掩码，找出哪些 token 应该由当前专家处理。
  - 遍历每个专家，只让它处理分配给自己的 token。
- 这样就实现了“不是所有专家都参与所有 token 计算”，从而达到稀疏激活的目的。

---

### 3. 负载均衡损失

这一阶段主要负责避免某些专家过忙、某些专家闲置：

1. 训练阶段
   - 需要统计实际负载情况。
   - 再根据路由结果计算 `aux_loss`。
   - 这个辅助损失会推动专家使用更均衡。

2. 推理阶段
   - 不需要计算 `aux_loss`。
   - 因为推理只关心前向结果，不再做参数更新。

```python
class MOEFeedForward(nn.Module):
    def __init__(self, config: MindConfig):
        super().__init__()
        self.config = config

        # 门控网络
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)

        # 专家集合
        self.experts = nn.ModuleList(
            [
                FeedForward(config,   
                intermediate_size=config.moe_intermediate_size)
                for _ in range(config.num_experts)
            ]
        )

        # 激活函数
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        batch_size, seq_len, hidden_size = x.shape

        # 1.【平铺数据】：把所有 Token 排成一长队，形状变为 (总Token数, hidden_size)
        x_flat = x.view(-1, hidden_size)

        # 2.【计算得分】：让 Gate 给每个 Token 分配专家的倾向性打分。
        # scores 形状（总 token 数，专家评分）
        scores = F.softmax(self.gate(x_flat), dim=-1)

        # 3.【Top-K 路由】：为每个 Token 挑出得分最高的 K 个专家。
        # topk_weight: 挑出的最高分数值，topk_idx: 对应的专家编号
        topk_weight, topk_idx = torch.topk(
            scores, k=self.config.num_experts_per_tok, dim=-1, sorted=False
        )

        # 4.【权重归一化】：把挑出来的 K 个权重重新按比例放大，使其和恢复到 1
        if self.config.norm_topk_prob:
            # topk_idx, topk_weight 形状为【num_tokens, k（得分高的前 k 专家）】
            topk_weight = topk_weight / (
	            topk_weight.sum(dim=-1, keepdim=True) + 1e-20
	            )

        # 5. 存放容器：创建一个和 x_flat 形状一样的全 0 矩阵，用来装各个专家的计算结果
        y = torch.zeros_like(x_flat)

        # 6.【稀疏计算核心】：遍历每一个专家，让它们只处理分配给自己的 Token。
        for i, expert in enumerate(self.experts):
            mask = （topk_idx == i）

            # any()：只要有一个是 True，就是 True。
            # 该序列中有专家被使用
            if mask.any():
                # token_idx 得到的是一个一维列表，比如 [1, 5, 12]，代表第 1、5、12 个词归当前专家管
                # .nonzero(): 找出上面所有标记为 True 的位置索引（行号），返回的是二维格式；.flatten() 变成一维数组
                token_idx = mask.any(dim=-1).nonzero().flatten()

                # pytorch 的布尔索引，只取 mask == True 的元素
                weight = topk_weight[mask].view(-1, 1)
                y.index_add(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))
            elif self.training:
                y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())

        # 7.【负载均衡损失 (Auxiliary Loss)】
        # self.config.router_aux_loss_coef 是一个调节系数
        # 训练阶段
        if self.training and self.config.router_aux_loss_coef > 0:
            # 统计实际负载：用 one_hot 把编号转成表格，再纵向求平均，算出每个专家实际分到了百分之几的 Token
            load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)

            # 计算偏心罚款：实际负载(load) * 预期负载(scores.mean) 求和，乘积越大代表越偏心，惩罚越重。
            self.aux_loss = (
                (load * scores.mean(0)).sum()
                * self.config.num_experts
                * self.config.router_aux_loss_coef
            )
        else:
            # 推理阶段：不需要算 Loss，生成一个与 scores 物理属性完全相同、形状为标量 0 的替身
            self.aux_loss = scores.new_zeros(1).squeeze()

        return y.view(batch_size, seq_len, hidden_size)
```

## 相关笔记

- **对比**：[[10-LLM-大语言模型/01-Architecture-模型架构/MiniMind架构图解析|MiniMind架构图解析]] — MiniMind 的稠密结构可作为理解 MoE 改造的基线。
- **系统影响**：[[20-AI-Infra-AI基础设施/01-Roadmaps-学习路线/LLM 推理加速与算子优化学习路线|推理加速与算子优化路线]] — MoE 的专家路由和稀疏激活会改变推理调度与算子需求。
