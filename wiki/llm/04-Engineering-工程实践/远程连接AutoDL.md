
# AutoDL 远程连接与环境初始化

关联阅读：可结合 [Python环境配置](Python%E7%8E%AF%E5%A2%83%E9%85%8D%E7%BD%AE.md) 理解；AutoDL 连接后仍需复现 Python 依赖与运行环境。


> [!summary]
> 这篇笔记分成两部分：
> - 先用 VS Code 连接 AutoDL 远程实例
> - 再用 `uv` 配置环境并启动训练

## 1. VS Code 连接 AutoDL

### 1.1 基本步骤

1. 在 VS Code 里添加远程主机。
2. 在终端输入 `ssh` 命令，并填写 password。
3. 在 VS Code 顶部弹窗里再次输入密码。
4. 连接成功后，安装 `uv`。
5. 配置环境；如果需要，可以使用无卡模式或克隆实例。

## 2. 环境初始化与训练

### 第一步：清理战场（切断代理 + 删除旧环境）

这一步的目的：

- 防止代理配置干扰网络。
- 删除之前失败残留的虚拟环境。

```bash
unset http_proxy && unset https_proxy && unset all_proxy
rm -rf .venv
```

### 第二步：极速打底（指定 Python 3.10 + 阿里云镜像下载基础包）

这一步的目的：

- 先把基础环境快速搭起来。
- 通过阿里云镜像提高依赖下载速度。

```bash
uv sync --python 3.10 --index-url https://mirrors.aliyun.com/pypi/simple/
```

### 第三步：安装 PyTorch（开启加速 + 下载 5090 对应版本）

这一步的目的：

- 挂载 AutoDL 官方加速通道。
- 重新安装适配当前 CUDA 版本的 `torch`。

> [!note]
> 这一步通常会下载一个较大的文件（约 3GB），可以放在后台等待完成。

```bash
source /etc/network_turbo
uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/cu128
```

### 第四步：启动训练

当前面的环境都准备好后，一定要切换回 **有卡模式**
就可以直接运行训练脚本：

```bash
.venv/bin/python trainer/train_pretrain.py
```

## 3. 最小执行顺序

如果只想快速照着做，可以按这个顺序执行：

1. 连接 AutoDL 远程实例。
2. 安装 `uv`。
3. 清理旧环境。
4. 同步基础依赖。
5. 安装 PyTorch。
6. 启动训练。

## 相关笔记

- **应用**：[Python环境配置](Python%E7%8E%AF%E5%A2%83%E9%85%8D%E7%BD%AE.md) — AutoDL 连接后仍需复现 Python 依赖与运行环境。
- **基础设施**：[vLLM／SGLang 路线](../../ai-infra/01-Roadmaps-%E5%AD%A6%E4%B9%A0%E8%B7%AF%E7%BA%BF/%E8%B7%AF%E7%BA%BF.md) — AutoDL 提供学习 vLLM、SGLang 和 GPU 推理所需的远程资源。
