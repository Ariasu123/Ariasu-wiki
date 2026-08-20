在大模型生命周期中，`eval.py`（推理侧）的核心任务只有一个：**把死板的参数权重，变成一个能听懂人话、能记住上下文的智能大脑。**

## 1. 初始化 Tokenizer 和模型

关联阅读：可结合 [SFT 监督微调](../02-Pretraining-%E9%A2%84%E8%AE%AD%E7%BB%83/SFT%20%E7%9B%91%E7%9D%A3%E5%BE%AE%E8%B0%83.md) 理解；Eval 用来判断 SFT 后的指令遵循和输出质量。


- **Tokenizer（分词器）**：大模型的“字典”。负责把人类的文本切碎，翻译成模型认识的数字 ID。
    
- **Model（模型架构）**：大模型的“大脑”。初始化时需要严格对齐训练时的超参数（如 `hidden_size`, `num_hidden_layers`，以及是否启用了 `use_moe` 混合专家架构）。如果不指定到对应的设备（`.to("cuda")`），数据和模型不在同一个地方，程序会直接报错。
    

## 2. 兼容本地原生加载与Hugging Face 加载

为了兼容不同的工程需求，加载权重通常要做 `if...else` 分流：

- **本地原生权重（`torch.load` + `load_state_dict`）**：
    
    - 通常是自己从零训练（Pretrain/SFT）保存的纯净参数字典（`.pth`）。
        
    - **高阶操作**：在这种模式下，可以非常灵活地进行“外挂操作”，比如通过 `apply_lora` 和 `load_lora` 动态注入特定领域的 LoRA 权重，让模型瞬间切换人设（如医疗专家）。
        
- **Hugging Face 加载（`from_pretrained`）**：
    
    - 工业标准格式，通常包含完整的 `config.json` 等配置文件，一行代码即可自动完成架构构建和权重赋值。
        

## 3. 交互分发：自动测试 vs 手动输入模式

- **自动测试（Batch Prompts）**：跑预设好的 prompt 列表（如“什么是光合作用”、“写个 Python 函数”），用来快速验证模型能力是否退化。
    
- **手动输入（Interactive CLI）**：进入类似终端的死循环 `while` 或无限 `iter`，等待用户输入，模拟真实的 ChatGPT 聊天窗口。

## 4. 记忆管理：维护多轮对话历史

大模型本身是没有记忆的（Stateless），必须靠代码在外部手动维护“记忆”。

- **实现原理**：维护一个 `conversation` 列表，把每一次的用户提问和 AI 回答按 `{"role": "...", "content": "..."}` 的格式追加进去。
    
- **显存保护机制**：通过 `conversation[-args.historys :]` 截取最近的 N 轮对话。防止历史记录无限变长，最终撑爆模型的最大上下文窗口（Context Window）或耗尽显存。

## 5. 构造 Chat Template / Pretrain 输入

这是决定模型“怎么说话”的最关键一步！

- **Pretrain 模式（无脑接话茬）**：
    
    - **拼接逻辑**：`<bos> + 你的输入`。
        
    - **效果**：模型不知道这是对话，只会把你输入的话当成一篇文章的开头，继续往下续写。
        
- **Chat/SFT 模式（照着剧本念台词）**：
    
    - **拼接逻辑**：调用 `tokenizer.apply_chat_template()`。
        
    - **效果**：把历史记录和当前问题包装成带特殊标记的剧本（如 `<|user|>问题<|im_end|>\n<|assistant|>`）。
        
    - **关键触发器（`add_generation_prompt=True`）**：在组装好的文本最后，强行加上 `<|assistant|>` 这个半截标记。相当于导演喊 "Action!"，明确提示模型：人类的话说完了，现在立刻以助手的身份开始输出！        

## 6. 张量化与数据搬运

- **分词（Tokenize）**：将上面组装好的长字符串 `inputs` 转换为数字矩阵。
    
- **关键参数**：`return_tensors="pt"`（返回 PyTorch 张量），`truncation=True`（超长自动截断）。
    
- **设备同步**：通过 `.to(args.device)` 将张量搬运到 GPU 显存中，准备交由模型计算。

## 7. 调用 `generate` 做自回归生成

这是推理的核心引擎。大模型每算出一个新词，就把这个新词加到已知条件里，再去算下一个词，循环往复。

- **超参数控制**：
    
    - `max_new_tokens`：最多允许生成多少个新词，防止模型像复读机一样死循环无限输出。
        
    - `temperature`：温度值（0~1）。越低越严谨客观，越高越发散且具创造力。
        
    - `top_p`：核采样。截断掉那些概率极低的离谱词汇，保证生成的合理性。
        
- **打字机特效（Streamer）**：通过传入 `TextStreamer`，拦截模型每一次循环生成的单个 Token 并立即打印到屏幕，消除人类用户的等待焦虑。
    

## 8.截取新生成内容并写回对话历史

- **为什么需要截取？** `model.generate` 默认返回的是 **“完整的输入序列 + 新生成的输出序列”**。
    
- **代码动作**：
    
    1. 通过切片 `generated_ids[0][len(inputs["input_ids"][0]) :]`，精确剔除掉前面的 prompt，只保留纯粹的 AI 新回答。
        
    2. 调用 `tokenizer.decode(..., skip_special_tokens=True)` 将数字 ID 翻译回人类能看懂的中文。
        
    3. 将这段纯文本组装成 `{"role": "assistant", "content": response}`，追加到 `conversation` 列表中。至此，一次完整的对话闭环结束，准备迎接用户的下一个问题。

## 相关笔记

- **验证**：[SFT 监督微调](../02-Pretraining-%E9%A2%84%E8%AE%AD%E7%BB%83/SFT%20%E7%9B%91%E7%9D%A3%E5%BE%AE%E8%B0%83.md) — Eval 用来判断 SFT 后的指令遵循和输出质量。
- **项目应用**：MiniMind 项目问答 — MiniMind 项目需要用推理脚本和指标验证训练阶段成果。
