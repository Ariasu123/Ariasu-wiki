

## Pretrain 核心模块

关联阅读：可结合 [模型权重拼接与 Hugging Face 封装](%E6%A8%A1%E5%9E%8B%E6%9D%83%E9%87%8D%E6%8B%BC%E6%8E%A5%E4%B8%8E%20Hugging%20Face%20%E5%B0%81%E8%A3%85.md) 理解；训练数据管线产出的模型需要标准化保存和加载。


### `train_epoch` 

**取数据 -> 算学习率 -> 前向传播 -> 反向传播 -> 梯度累积 -> 更新参数 -> 打日志 -> 存 checkpoint**

---

### 1. 数据准备与学习率更新

这一阶段主要负责：

- 从 `loader` 中依次取出 `input_ids`、`labels`、`attention_mask`。
- 把这些张量移动到训练设备上。
- 调用 `get_lr(...)` 按当前训练进度计算学习率。
- 遍历 `optimizer.param_groups`，把当前 step 对应的 `lr` 写回优化器。
---

### 2. 前向传播与 Loss 计算

这一阶段主要负责：

- 进入 `autocast_ctx` 混合精度上下文，减少显存占用并提升训练速度。
- 调用 `model(input_ids, labels=labels, attention_mask=attention_mask)` 做前向传播。
- 得到输出结果 `res`。
- 使用 `res.loss + res.aux_loss` 作为总训练目标。
- 在梯度累积场景下，先执行 `loss = loss / args.accumulation_steps`，保证梯度量级正确。
---

### 3. 反向传播与梯度累积

这一阶段主要负责：

- 调用 `scaler.scale(loss).backward()` 进行反向传播。
- 对 float16 而言，直接 backward 有时会因为梯度太小而下溢；
	scaler 会先把 loss 放大，再在更新前适时还原。
- 在混合精度训练里，通过 `scaler` 避免梯度下溢。
- 只有当 `step % args.accumulation_steps == 0` 时，才真正进入一次参数更新；其他 step 只做梯度累积。

---

### 4. 参数更新

当满足梯度累积更新条件后，才会进入真正的参数更新流程：

1. 调用 `scaler.unscale_(optimizer)`，把放大的梯度还原回真实值。
2. 使用 `torch.nn.utils.clip_grad_norm_(...)` 做梯度裁剪，防止梯度爆炸。
3. 调用 `scaler.step(optimizer)` 执行优化器更新。
4. 调用 `scaler.update()` 更新缩放因子。
5. 使用 `optimizer.zero_grad(set_to_none=True)` 清空当前累积窗口的梯度。

---

### 5. 日志记录与模型保存

这一阶段主要负责训练过程的观测和持久化：

- 按 `log_interval` 周期性打印当前 `loss`、`lr` 和预计剩余时间。
- 如果启用了 `wandb`，同步记录实验指标。
- 按 `save_interval` 或在 epoch 最后一步时保存模型。
- 主进程会保存：
  - 半精度权重文件：用于推理加载。
  - 完整 checkpoint：用于断点续训。

```python
def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    """
    参数:
        loader (DataLoader): 每次迭代会返回一个 batch，
            返回 `(input_ids, labels, attention_mask)` 三个张量。
        iters (int):
            当前 epoch 代表的是“这一轮训练按计划会执行多少步”。
    """
    start_time = time.time()

    for step, (input_ids, labels, attention_mask) in enumerate(
        loader, start=start_step + 1
    ):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        attention_mask = attention_mask.to(args.device)

        lr = get_lr(
	        epoch * iters + step, 
	        args.epochs * iters, 
	        args.learning_rate
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # 在 autocast 上下文中执行前向计算。
        with autocast_ctx:
            res = model(
	            input_ids, labels=labels, attention_mask=attention_mask
            )
            loss = res.loss + res.aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0:
            # 在做梯度裁剪之前，必须先把被 scaler 放大的梯度还原回真实值。
            # 否则裁剪阈值会失真，导致裁剪行为不符合预期。
            scaler.unscale_(optimizer)

            # 当梯度范数超过阈值时，会被整体缩放到 `args.grad_clip` 附近。
            torch.nn.utils.clip_grad_norm_（
				model.parameters(), 
				args.grad_clip
			)

            # 如果当前梯度正常，则执行一次优化器更新。
            scaler.step(optimizer)

            # 根据当前梯度是否稳定，动态调整 loss scale。
            # 这一步对 float16 混合精度训练尤其重要。
            scaler.update()

            # 清空上一轮累计的梯度，为下一次累积窗口做准备。
            # `set_to_none=True` 通常会比置零更省内存，也更高效。
            optimizer.zero_grad(set_to_none=True)
```
## Dataset

### `PretrainDataset` 

**读取数据 -> tokenize -> 拼接 BOS/EOS -> padding -> 构造 labels -> 构造 attention_mask -> 返回样本**

---

### 1. 数据读取与初始化

- 在 `__init__` 中保存 `tokenizer` 和 `max_length`。
- 使用 `load_dataset("json", data_files=data_path, split="train")` 读取 json 数据。
- 把读取结果保存到 `self.samples`。

---

### 2. 单条样本处理

- 按索引从 `self.samples` 里取出一条原始样本。
- 调用 `tokenizer` 把文本编码成 token id。
- 预留前后两个位置给 `BOS` 和 `EOS`。
- 在 token 序列前后拼接 `BOS` 和 `EOS`。
- 用 `PAD` 把序列补到固定长度，得到 `input_ids`。

---

### 3. 标签与掩码构造

- 复制 `input_ids` 得到 `labels`。
- 把 padding 位置改成 `-100`，使 `CrossEntropyLoss` 自动忽略这些位置。
- 构造 `attention_mask`，其中有效 token 为 `1`，padding 为 `0`。
- 返回 `input_ids, labels, attention_mask`。

---

### 4. Dataset 和 DataLoader 的关系

- `Dataset` 定义“单条样本怎么取”。
- `__len__` 告诉程序数据集有多少条样本。
- `__getitem__` 告诉程序给定索引时该返回什么内容。
- `DataLoader` 会不断调用 `Dataset.__getitem__`，再把多条样本组成一个 batch。
---

### 5. 代码细节说明

1. `self.samples` 不是普通 Python `list`，而是 Hugging Face 的 `Dataset` 对象。
2. `sample = self.samples[index]` 取出来的通常是一条字典数据，例如：`{"text": "hello world"}`。
3. `(input_ids != self.tokenizer.pad_token_id).long()` 会先得到布尔张量，再把 `True / False` 转成 `1 / 0`。
------

```python
class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset(
	        "json", 
	        data_files=data_path, 
	        split="train"
		)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        tokens = self.tokenizer(
            str(sample["text"]),
            add_special_tokens=False,
            max_length=self.max_length - 2,
            truncation=True,
        ).input_ids

        tokens = [self.tokenizer.bos_token_id] + 
			     tokens + 
			     [self.tokenizer.eos_token_id]

        input_ids = tokens + [self.tokenizer.pad_token_id] * (
            self.max_length - len(tokens)
        )
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100

        attention_mask = (
	        input_ids != self.tokenizer.pad_token_id
	        ).long()
        return input_ids, labels, attention_mask
```

## Utils

### `utils` 模块

**主进程控制 -> 日志输出 -> 学习率调度 -> 分布式初始化 -> 随机种子 -> checkpoint -> 模型初始化 -> 断点续训采样**

---

### 1. 主进程判断与日志输出

这一部分主要负责训练过程中的输出控制：

- `is_main_process()`：判断当前是不是主进程。
- `Logger(content)`：只在主进程打印日志，避免多卡训练时重复输出。

---

### 2. 学习率与训练环境初始化

这一部分主要负责训练开始前的基础准备：

- `get_lr(current_step, total_steps, lr)`：按训练进度计算动态学习率，通常使用余弦退火。
- `init_distributed_mode()`：初始化分布式训练环境，设置进程组和当前进程使用的 GPU。
- `setup_seed(seed)`：设置随机种子，尽量保证实验可复现。

---

### 3. 模型与 checkpoint 管理

这一部分主要负责训练状态和模型状态的管理：

- `lm_checkpoint(...)`：保存或加载训练检查点，包括模型参数、优化器状态、epoch 和 step 等信息，用于断点续训。
- `init_model(...)`：初始化模型和 tokenizer，并在需要时加载已有权重。

---

### 4. 断点续训相关采样

这一部分主要负责从中断位置继续训练：

- `SkipBatchSampler`：自定义采样器，用于跳过前面已经训练过的 batch，常见于断点续训场景。

## 相关笔记

- **工程出口**：[模型权重拼接与 Hugging Face 封装](%E6%A8%A1%E5%9E%8B%E6%9D%83%E9%87%8D%E6%8B%BC%E6%8E%A5%E4%B8%8E%20Hugging%20Face%20%E5%B0%81%E8%A3%85.md) — 训练数据管线产出的模型需要标准化保存和加载。
- **理论**：[LLM预训练底层机制](../02-Pretraining-%E9%A2%84%E8%AE%AD%E7%BB%83/LLM%20%E9%A2%84%E8%AE%AD%E7%BB%83%E5%BA%95%E5%B1%82%E6%9C%BA%E5%88%B6.md) — 代码笔记是预训练数据与优化机制的具体实现。
- **工程基础**：[PyTorch及相关方法](../04-Engineering-%E5%B7%A5%E7%A8%8B%E5%AE%9E%E8%B7%B5/PyTorch%20%E5%8F%8A%E7%9B%B8%E5%85%B3%E6%96%B9%E6%B3%95.md) — Dataset、DataLoader 和张量操作都建立在 PyTorch API 上。
- **流程衔接**：[SFT 监督微调](../02-Pretraining-%E9%A2%84%E8%AE%AD%E7%BB%83/SFT%20%E7%9B%91%E7%9D%A3%E5%BE%AE%E8%B0%83.md) — 预训练与 SFT 都需要构造输入、标签和 loss mask。
