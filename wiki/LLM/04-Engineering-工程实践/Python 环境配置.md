
## VS Code 插件推荐

关联阅读：可结合 [PyTorch及相关方法](PyTorch%20%E5%8F%8A%E7%9B%B8%E5%85%B3%E6%96%B9%E6%B3%95.md) 理解；环境配置的主要使用者之一是 PyTorch 训练代码。


- `Chinese`：提供 VS Code 中文界面。
- `Dracula`：提供暗色主题。
- `Material Icon Theme`：提供文件和文件夹图标。
- `Path Intellisense`：提供路径自动补全。
- `Prettier`：负责代码格式化。
- `Python`：提供 Python 语法高亮、运行、调试等基础开发支持。
- `Ruff`：提供 Python 代码检查和规范支持。
## 1. uv
### 1.1 uv 是什么

`uv` 是一个现代 Python 工具链，核心目标是快、统一、工程化。它把 Python 项目里原本分散的几类工作收拢到一起：
- 管理 Python 版本
- 创建虚拟环境
- 安装和解析依赖
- 维护锁文件
- 运行项目命令
- 运行或安装工具型包
### 1.2 uv 常用操作

#### 1. 初始化项目
```bash
uv init
```
#### 2. 安装和管理 Python 版本
```bash
uv python list
```
安装某个版本：
```bash
uv python install 3.12
```
将当前项目固定到某个 Python 版本：
```bash
uv python pin 3.12
```
#### 3. 创建虚拟环境
```bash
uv venv
```
#### 4. 添加依赖
```bash
uv add 
```
#### 5. 安装项目环境
```bash
uv sync
```
#### 6. 运行或安装工具
全局安装某个工具命令：
```bash
uv tool install ruff
uv tool list
```
激活：source .venv/bin/activate # 激活 (Mac/Linux)
### 1.3 什么时候用 uv，什么时候用 conda
![](../_assets/Engineering-%E5%B7%A5%E7%A8%8B%E5%AE%9E%E8%B7%B5/uv-%20conda%E6%AF%94%E8%BE%83.png)

## 2. pyproject.toml
### 2.1 pyproject.toml 是什么
`pyproject.toml` 是现代 Python 项目的核心配置文件。它通常承担三类职责：
- 描述项目本身
- 描述依赖与构建方式
- 给各种工具提供统一配置入口
常见理解方式是：
- `requirements.txt` 更像“安装清单”
- `pyproject.toml` 更像“项目说明书 + 依赖声明 + 工具配置中心”
### 2.2 一个典型的 pyproject.toml 结构
下面是一个常见示例：

```toml

[build-system]

requires = ["hatchling>=1.25.0"]

build-backend = "hatchling.build"


[project]

name = "demo-app"

version = "0.1.0"

description = "A demo Python application"

readme = "README.md"

requires-python = ">=3.11"

dependencies = [

"httpx>=0.27.0",

"rich>=13.9.0",

]

[project.optional-dependencies]

cli = [

"typer>=0.12.0",

]

[project.scripts]

demo-app = "demo_app.main:main"

[dependency-groups]

dev = [

"pytest>=8.0.0",

"ruff>=0.6.0",

]

[tool.ruff]

line-length = 88

target-version = "py311"


[tool.ruff.lint]

select = ["E", "F", "I", "UP"]


[tool.ruff.format]

quote-style = "double"

```
### 2.3 关键部分解释

#### 1. `[build-system]`

如果你只是写内部应用，前期可能感觉不到它的重要性；但只要你要打包、发布、构建 wheel，它就很关键。
#### 2. `[project]`
这是项目元数据的主体区域，常见字段有：
- `name`：项目名
- `version`：版本号
- `description`：描述
- `readme`：说明文档
- `requires-python`：支持的 Python 版本
- `dependencies`：运行时依赖

这里的 `dependencies` 指的是项目运行真正需要的依赖。
#### 3. `[tool.xxx]`
这部分是给具体工具留的位置。不同工具会在这里读取自己的配置。
- `[tool.ruff]`
这也是 `pyproject.toml` 很有价值的地方：一个文件集中管理项目级配置。
## 3. Ruff
### 3.1 Ruff 是什么
`ruff` 是一个高性能的 Python 代码检查和格式化工具。
它常被用来处理这些事情：
- 发现语法或导入错误
- 发现未使用变量、未使用导入
- 排序 import
- 自动修复一部分问题
- 格式化代码
### 3.2 Ruff 常用操作

|任务|命令|
|---|---|
|**格式化我的代码**|`ruff format .`|
|**检查代码问题并自动修复**|`ruff check . --fix`|
|**在 CI/CD 中验证格式**|`ruff format . --check`|
|**在 CI/CD 中检查代码质量**|`ruff check .`|
|**开发时实时检查**|`ruff check . --watch`|
|**搞不清某条规则是什么意思**|`ruff rule <RULE_CODE>`|
|**清理缓存**|`ruff cache clean`|
|**终极清理**|`ruff format . && ruff check . --fix`|

### 3.3 Ruff 常见配置
Ruff 通常写在 `pyproject.toml` 里：
```toml
[tool.ruff]
line-length = 88
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]
ignore = []

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
```
这些配置的常见含义：
- `line-length`：每行最大长度
- `target-version`：按哪个 Python 版本规则检查
- `select`：启用哪些规则族
- `ignore`：忽略哪些规则
几个常见规则族：
- `E` / `W`：基础风格问题
- `F`：Pyflakes 类问题，比如未使用变量、未定义名称
- `I`：import 排序
- `UP`：推荐升级到更现代的 Python 写法
- `B`：常见 bug 风险

### 3.4 Ruff 的实践建议
一个实用的最小策略是：
```toml
[tool.ruff]
line-length = 88
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP"]
```
这样已经能覆盖大量日常问题。
建议：
- 先让 `ruff` 接管 import 排序和基础 lint
- 再决定是否让它接管格式化
- 团队统一后，把检查接到 CI

## 4. Rich：漂亮的 traceback

### 4.1 Rich 是什么
`rich` 是一个终端美化库，可以让命令行输出更清晰，常见能力包括：
- 彩色文本
- 表格
- 面板
- 日志
- traceback
### 4.2 为什么 Rich 的 traceback 好用
默认 Python traceback 能用，但展示通常比较“硬”。`rich` 的 traceback 更适合阅读，因为它可以：
- 高亮代码
- 更清楚地标出出错位置
- 更好地区分调用栈层级
- 可选展示局部变量
### 4.3 最简单的用法
先安装：
```bash
uv add rich
```
然后在程序入口尽早安装 traceback handler：
```python
from rich.traceback import install

install()
```
### 4.4 常用参数
#### `show_locals=True`
显示局部变量，非常适合调试：
```python
from rich.traceback import install

install(show_locals=True)
```
但要注意：局部变量里如果有密码、token、用户隐私数据，就不适合在生产环境随便打开。

#### `suppress=[...]`
隐藏某些库内部的大段栈信息，让 traceback 更聚焦在你的业务代码上：
```python
import click
from rich.traceback import install

install(show_locals=True, suppress=[click])
```

## 相关笔记

- **用途**：[PyTorch及相关方法](PyTorch%20%E5%8F%8A%E7%9B%B8%E5%85%B3%E6%96%B9%E6%B3%95.md) — 环境配置的主要使用者之一是 PyTorch 训练代码。
- **基础**：Python3_ACM输入输出与手撕模板 — Python 环境配置支撑算法练习和面试运行。
