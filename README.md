# Gemma-4-E4B 本地部署 — 工具调用与技能系统

基于 [llama.cpp](https://github.com/ggml-org/llama.cpp) 部署 [Gemma-4-E4B-Uncensored-HauhauCS-Aggressive](https://huggingface.co/HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive) 模型，集成 OpenAI Function Calling 工具调用、skill.md 自定义技能、MCP 协议扩展与 AGENTS.md 系统提示词。

---

## 目录

- [特性概览](#特性概览)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [部署步骤](#部署步骤)
- [架构说明](#架构说明)
- [AGENTS.md 系统提示词](#agentsmd-系统提示词)
- [工具调用系统](#工具调用系统)
  - [内置工具](#内置工具)
  - [自定义工具（Python 类）](#自定义工具python-类)
  - [skill.md 文件技能](#skillmd-文件技能)
  - [MCP 协议工具](#mcp-协议工具)
- [API 接口文档](#api-接口文档)
- [交互式聊天命令](#交互式聊天命令)
- [配置参考](#配置参考)
- [示例代码](#示例代码)
- [常见问题](#常见问题)

---

## 特性概览

| 特性 | 说明 |
|------|------|
| **GPU 推理** | 基于 llama.cpp CUDA 后端，RTX 4070 约 67 tokens/s |
| **工具调用** | OpenAI Function Calling 协议，自动循环执行（最多 10 轮） |
| **7 个内置工具** | calculator / file_ops / datetime / system_info / code_exec / shell_exec / web_fetch |
| **skill.md 扩展** | 在 `skills/` 目录放置 Markdown 文件即可新增工具，支持 YAML 元数据 + Python 代码 |
| **MCP 协议** | 支持 stdio / HTTP 两种传输，自动发现并注册远程工具 |
| **AGENTS.md** | Markdown 格式的系统提示词，控制模型行为与工具调用策略 |
| **OpenAI 兼容 API** | FastAPI 代理服务器，提供 `/v1/chat/completions` 等标准接口 |
| **多模态** | 原生支持图片/音频输入（需 mmproj 文件） |

---

## 环境要求

| 项目 | 最低要求 | 推荐 |
|------|---------|------|
| GPU | NVIDIA GPU, 6 GB+ VRAM | RTX 4070+ (8 GB VRAM) |
| Python | 3.10+ | 3.12 |
| CUDA Driver | 12.1+ | 12.4+ |
| 磁盘空间 | 7 GB | 10 GB |
| 内存 | 8 GB | 16 GB |

> 本项目使用 llama.cpp 官方预编译二进制，**无需**安装 CUDA Toolkit 或 C++ 编译器。

---

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 一键下载（模型 + llama.cpp 二进制）
python start.py download

# 3. 启动交互式聊天
python chat.py

# 4. 或启动 API 服务器
python start.py serve
```

---

## 部署步骤

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

| 依赖包 | 用途 |
|--------|------|
| `requests` | HTTP 请求（与 llama-server 通信） |
| `fastapi` + `uvicorn` | OpenAI 兼容 API 代理服务器 |
| `openai` | OpenAI 兼容客户端 |
| `pydantic` | 数据模型校验 |
| `huggingface_hub` | 从 HuggingFace 下载模型 |

### 2. 下载模型与二进制

```bash
python start.py download
```

自动下载以下文件：

| 文件 | 大小 | 说明 |
|------|------|------|
| `llama-server.exe` + DLLs | ~223 MB | llama.cpp CUDA 12.4 预编译二进制 |
| `Gemma-4-E4B-...Q4_K_M.gguf` | ~5.0 GB | 主模型（Q4_K_M 量化） |
| `mmproj-...f16.gguf` | ~945 MB | 多模态投影（图片/音频） |

> 设置 HF Token 可加速下载：
> ```bash
> set HF_TOKEN=your_token_here
> python start.py download
> ```

### 3. 验证部署

```bash
python start.py info
```

输出示例：
```
System Information:
  llama-server: ...\llama-cpp-bin\llama-server.exe (exists: True)
  Model file: ...\models\Gemma-4-E4B-...Q4_K_M.gguf (exists: True)
  Model size: 4.95 GB
  mmproj file: ...\models\mmproj-...f16.gguf (exists: True)
  mmproj size: 944.4 MB
  CUDA available: True
  GPU: NVIDIA GeForce RTX 4070 Laptop GPU
```

### 4. 启动服务

**交互式聊天**（推荐首次使用）：
```bash
python chat.py
```

**API 服务器**：
```bash
python start.py serve --host 127.0.0.1 --port 8080
```

**直接使用 llama-server**（不含工具调用层）：
```bash
llama-cpp-bin\llama-server.exe ^
  -m models\Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf ^
  --mmproj models\mmproj-Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-f16.gguf ^
  --host 127.0.0.1 --port 8081 -c 8192 --jinja -ngl 99
```

---

## 架构说明

### 目录结构

```
gemma4-e4b/
├── AGENTS.md                        # 系统提示词（控制模型行为）
├── mcp_config.json                  # MCP 服务器配置
├── models/                          # 模型文件
│   ├── *.gguf                       # 主模型
│   └── mmproj-*.gguf                # 多模态投影
├── skills/                          # skill.md 自定义技能目录
│   ├── text_transform.md            # 文本转换工具
│   └── json_tool.md                 # JSON 处理工具
├── llama-cpp-bin/                   # llama.cpp 预编译二进制
│   ├── llama-server.exe             # 推理服务器
│   ├── llama-cli.exe                # 命令行推理
│   ├── ggml-cuda.dll                # CUDA 后端
│   └── cudart64_12.dll              # CUDA 运行时
├── src/
│   ├── config.py                    # 全局配置
│   ├── server.py                    # FastAPI 代理服务器
│   ├── skills/
│   │   ├── base.py                  # Skill 抽象基类
│   │   ├── registry.py              # 技能注册中心
│   │   ├── skill_md.py              # skill.md 文件解析器
│   │   └── builtin/                 # 内置技能
│   │       ├── calculator.py        # 计算器
│   │       ├── file_ops.py          # 文件操作 / 日期时间 / 系统信息
│   │       └── code_exec.py         # 代码执行 / Shell / 网页抓取
│   └── tools/
│       ├── client.py                # GemmaClient + LlamaServerManager + MCPManager
│       ├── executor.py              # 工具执行引擎
│       └── mcp_client.py            # MCP 协议客户端
├── chat.py                          # 交互式聊天入口
├── start.py                         # 统一 CLI 入口
├── test.py                          # 测试脚本
└── requirements.txt                 # 依赖列表
```

### 核心流程

```
用户请求 → FastAPI (8080) → GemmaClient → llama-server (8081) → GPU 推理
                                      ↑                              ↓
                                      │                     模型返回 tool_calls?
                                      │                    ╱ Yes          ╲ No
                                      │             ToolExecutor      返回文本
                                      │             执行 Skill              ↓
                                      │                  ↓            输出给用户
                                      └── 将结果追加到消息 ┘
                                           再次推理（最多 10 轮）
```

### 双端口架构

| 端口 | 服务 | 说明 |
|------|------|------|
| 8080 | FastAPI 代理 | 对外 API，处理工具调用循环 |
| 8081 | llama-server | 内部推理后端，不对外暴露 |

---

## AGENTS.md 系统提示词

`AGENTS.md` 是 Markdown 格式的系统提示词文件，在每次对话时自动加载为 system message，用于控制模型的行为和工具调用策略。

### 文件位置

项目根目录下的 `AGENTS.md`。

### 加载机制

`GemmaClient` 初始化时自动读取 `AGENTS.md` 内容作为系统提示词。如果文件不存在，使用内置的默认提示词。

### 文件结构

```markdown
# AGENTS — Gemma-4-E4B 系统指令

## 身份与角色
定义模型身份和基本定位

## 核心行为准则
1. 语言一致性 — 始终使用用户使用的语言回复
2. 工具优先 — 优先使用工具而非自行推测
3. 诚实透明 — 不编造工具结果
4. 安全意识 — 拒绝危险操作
5. 简洁高效 — 避免冗余信息

## 工具调用规范
定义何时调用工具、调用流程、注意事项

## 技能系统（Skills）
列出所有内置技能及使用方法

## 多轮对话 / 错误处理 / 限制与边界
...
```

### 自定义

直接编辑 `AGENTS.md` 即可修改模型行为。修改后重启服务生效。

---

## 工具调用系统

本系统采用 **OpenAI Function Calling 协议**。模型在需要调用工具时生成 `tool_calls` 字段，系统自动执行工具并将结果返回模型，模型根据结果生成最终回答。整个过程对用户透明。

### 内置工具

| 工具名 | 描述 | 参数 |
|--------|------|------|
| `calculator` | 数学表达式计算，支持三角函数、对数等 | `expression` (string, required): 数学表达式，如 `"sin(pi/4) + 2^10"` |
| `file_ops` | 文件读写、目录列表、文件存在检查 | `action` (string, required): `"read"` / `"write"` / `"list"` / `"exists"`<br>`path` (string, required): 文件或目录路径<br>`content` (string): 写入内容（仅 write 操作） |
| `datetime` | 获取当前日期时间 | `action` (string, required): `"now"` 返回 ISO 格式 / `"format"` 自定义格式<br>`format_str` (string): strftime 格式字符串，默认 `"%Y-%m-%d %H:%M:%S"` |
| `system_info` | 查询系统信息 | `info_type` (string, required): `"basic"` 基本信息 / `"memory"` 内存 / `"disk"` 磁盘 |
| `code_exec` | 执行 Python 代码并返回输出 | `code` (string, required): Python 代码<br>`timeout` (integer): 超时秒数，默认 30 |
| `shell_exec` | 执行 Shell 命令并返回输出 | `command` (string, required): Shell 命令<br>`timeout` (integer): 超时秒数，默认 30 |
| `web_fetch` | 抓取网页内容 | `url` (string, required): 目标 URL<br>`max_length` (integer): 最大返回长度，默认 5000 |

### 自定义工具（Python 类）

继承 `Skill` 基类并注册到全局 Registry：

```python
from src.skills.base import Skill
from src.skills.registry import get_registry

class TranslatorSkill(Skill):
    name = "translator"
    description = "Translate text between languages"
    parameters = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Text to translate"},
            "target_lang": {"type": "string", "description": "Target language code"}
        },
        "required": ["text", "target_lang"]
    }

    def execute(self, text: str = "", target_lang: str = "en", **kwargs) -> str:
        # 在此实现翻译逻辑
        return f"[Translation to {target_lang}]: {text}"

# 注册
registry = get_registry()
registry.register(TranslatorSkill())
```

**Skill 基类接口**：

| 属性/方法 | 类型 | 说明 |
|-----------|------|------|
| `name` | `str` | 工具唯一标识，对应 OpenAI tool 的 `function.name` |
| `description` | `str` | 工具描述，供模型理解何时调用 |
| `parameters` | `dict` | JSON Schema 格式的参数定义 |
| `execute(**kwargs) -> str` | 方法 | 执行工具逻辑，返回字符串结果 |
| `to_openai_tool() -> dict` | 方法 | 转换为 OpenAI Tool 格式 |

### skill.md 文件技能

在 `skills/` 目录下放置 Markdown 文件即可自动注册新工具。系统启动时自动扫描并加载所有 `.md` 文件。

#### 文件格式

```markdown
---
name: my_tool
description: "工具描述（含冒号时必须加引号）"
parameters:
  type: object
  properties:
    input:
      type: string
      description: 输入参数
  required:
    - input
---

# My Tool

工具的详细说明。

## Implementation

​```python
def execute(input: str = "", **kwargs):
    return f"Result: {input}"
​```
```

#### 格式说明

| 部分 | 是否必需 | 说明 |
|------|---------|------|
| YAML frontmatter (`---` 之间) | 推荐 | 定义 `name`、`description`、`parameters` |
| `## Implementation` 代码块 | 必需 | 必须包含 `def execute(...)` 函数 |
| Markdown 正文 | 可选 | 供人阅读的文档，不影响工具功能 |

#### 可用标准库

skill.md 代码中可直接使用以下 Python 标准库（无需 import）：

`json`, `math`, `base64`, `datetime`, `collections`, `itertools`, `os`, `sys`, `re`, `Path`

#### 注意事项

- YAML `description` 字段含冒号时**必须加引号**，否则解析失败
- `execute` 函数的参数名必须与 `parameters.properties` 中的键名一致
- 建议为 `execute` 参数提供默认值，增强容错性
- 运行时使用 `/reload` 命令可热加载新增的 skill.md 文件

#### 示例：text_transform.md

```markdown
---
name: text_transform
description: "Transform text with various operations"
parameters:
  type: object
  properties:
    text:
      type: string
      description: The input text
    operation:
      type: string
      description: "uppercase, lowercase, reverse, base64_encode, base64_decode, word_count"
  required:
    - text
    - operation
---

## Implementation

​```python
import base64

def execute(text: str = "", operation: str = "lowercase", **kwargs):
    ops = {
        "uppercase": lambda t: t.upper(),
        "lowercase": lambda t: t.lower(),
        "reverse": lambda t: t[::-1],
        "base64_encode": lambda t: base64.b64encode(t.encode()).decode(),
        "base64_decode": lambda t: base64.b64decode(t.encode()).decode(),
        "word_count": lambda t: str(len(t.split())),
    }
    fn = ops.get(operation)
    if fn is None:
        return f"Unknown operation: {operation}"
    return fn(text)
​```
```

### MCP 协议工具

系统支持 [Model Context Protocol (MCP)](https://modelcontextprotocol.io/)，可连接远程 MCP 服务器并自动注册其提供的工具。

#### 传输方式

| 方式 | 适用场景 | 配置字段 |
|------|---------|---------|
| **stdio** | 本地 MCP 服务器（通过子进程启动） | `command`: 启动命令列表 |
| **HTTP** | 远程 MCP 服务器 | `url`: 服务器 URL |

#### 配置文件

系统已预配置以下MCP服务器（修改 `mcp_config.json` 即可启用）：

```json
{
  "mcpServers": {
    "fetch": {
      "command": ["uvx", "--from", "mcp-server-fetch", "mcp-server-fetch"],
      "description": "Web fetcher - fetches URLs and extracts content as markdown"
    }
  }
}
```

> **注意**：使用 `uvx` 方式需要确保 `uv` 工具已安装（`pip install uv`）。也可使用 pip 安装：`pip install mcp-server-fetch`，然后将 `command` 改为 `["python", "-m", "mcp_server_fetch"]`。

| MCP服务器 | 工具 | 说明 |
|-----------|------|------|
| `fetch` (mcp-server-fetch) | `fetch` | 抓取URL并提取为markdown格式，支持分块读取 |

系统启动时自动连接配置中的 MCP 服务器，发现并注册其工具。连接失败的服务器会被跳过并打印警告。

#### 动态连接

运行时通过 API 或聊天命令动态连接 MCP 服务器：

**API 方式**：
```bash
curl -X POST http://127.0.0.1:8080/v1/mcp/connect \
  -H "Content-Type: application/json" \
  -d '{"name": "my_server", "url": "http://localhost:3001/mcp"}'
```

**聊天命令**：
```
/mcp my_server http://localhost:3001/mcp
/mcp fetch uvx --from mcp-server-fetch mcp-server-fetch
```

#### MCP 协议流程

```
GemmaClient                    MCP Server
    │                              │
    │── initialize ───────────────→│  协议版本 & 能力协商
    │←─ result (capabilities) ────│
    │                              │
    │── tools/list ───────────────→│  发现可用工具
    │←─ result (tools[]) ─────────│
    │                              │
    │── tools/call ───────────────→│  调用工具
    │←─ result (content[]) ───────│
```

---

## API 接口文档

启动 API 服务器（`python start.py serve`）后，提供以下接口：

### Chat Completions

```
POST /v1/chat/completions
```

**请求体**：

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model` | string | `"gemma-4-e4b"` | 模型标识 |
| `messages` | array | — | 消息列表，格式同 OpenAI API |
| `tools` | array | `null` | 工具列表，`null` 时使用所有已注册工具 |
| `temperature` | float | `1.0` | 采样温度 |
| `top_p` | float | `0.95` | 核采样阈值 |
| `top_k` | int | `64` | Top-K 采样 |
| `max_tokens` | int | `2048` | 最大生成 token 数 |
| `stream` | bool | `false` | 是否流式输出 |

**请求示例**：

```bash
curl -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4-e4b",
    "messages": [{"role": "user", "content": "计算 123 * 456"}],
    "temperature": 0.7,
    "max_tokens": 200
  }'
```

**流式响应**（`stream: true`）：

返回 SSE 格式，每个 chunk 格式：
```
data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"delta":{"content":"..."}}]}

data: [DONE]
```

### 其他接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `GET /v1/models` | GET | 列出可用模型 |
| `GET /v1/models/{id}` | GET | 获取模型信息 |
| `GET /v1/tools` | GET | 列出所有已注册工具（含内置 + skill.md + MCP） |
| `POST /v1/mcp/connect` | POST | 动态连接 MCP 服务器 |
| `POST /v1/skills/reload` | POST | 重新加载 `skills/` 目录下的 skill.md 文件 |
| `GET /health` | GET | 健康检查 |

### MCP 连接接口

```
POST /v1/mcp/connect
```

**请求体**：

```json
{
  "name": "server_name",
  "command": ["uvx", "--from", "mcp-server-fetch", "mcp-server-fetch"],
  "url": "http://localhost:3001/mcp"
}
```

`command` 和 `url` 二选一。

**响应**：

```json
{
  "status": "ok",
  "tools_loaded": ["read_file", "write_file", "list_directory"]
}
```

---

## 交互式聊天命令

启动 `python chat.py` 后，支持以下命令：

| 命令 | 说明 |
|------|------|
| `/tools` | 列出所有已注册工具及其描述 |
| `/clear` | 清空对话历史 |
| `/quit` | 退出聊天 |
| `/raw` | 切换原始工具调用信息显示（调试用） |
| `/reload` | 重新加载 `skills/` 目录下的 skill.md 文件 |
| `/mcp <name> <cmd_or_url>` | 动态连接 MCP 服务器 |
| `/agents` | 显示当前 AGENTS.md 系统提示词内容 |

---

## 配置参考

所有配置集中在 `src/config.py`：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `SERVER_HOST` | `"127.0.0.1"` | FastAPI 代理服务器监听地址 |
| `SERVER_PORT` | `8080` | FastAPI 代理服务器端口 |
| `LLAMA_SERVER_PORT` | `8081` | llama-server 内部端口 |
| `N_CTX` | `8192` | 上下文窗口长度 |
| `N_BATCH` | `512` | 批处理大小 |
| `TEMPERATURE` | `1.0` | 默认采样温度 |
| `TOP_P` | `0.95` | 默认核采样阈值 |
| `TOP_K` | `64` | 默认 Top-K |
| `TOOL_CALL_MAX_ITERATIONS` | `10` | 工具调用最大循环次数 |

**关键路径配置**：

| 配置项 | 默认路径 | 说明 |
|--------|---------|------|
| `MODEL_FILE` | `models/Gemma-4-E4B-...Q4_K_M.gguf` | 主模型文件 |
| `MMPROJ_FILE` | `models/mmproj-...f16.gguf` | 多模态投影文件 |
| `AGENTS_MD` | `AGENTS.md` | 系统提示词文件 |
| `SKILLS_DIR` | `skills/` | skill.md 文件目录 |
| `MCP_CONFIG` | `mcp_config.json` | MCP 服务器配置 |

---

## 示例代码

### Python 客户端（OpenAI SDK）

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="not-needed")

# 简单对话
response = client.chat.completions.create(
    model="gemma-4-e4b",
    messages=[{"role": "user", "content": "你好！"}],
)
print(response.choices[0].message.content)

# 带工具调用（自动使用所有已注册工具）
response = client.chat.completions.create(
    model="gemma-4-e4b",
    messages=[{"role": "user", "content": "用 calculator 计算 sin(pi/4)"}],
    temperature=0.5,
)
print(response.choices[0].message.content)
```

### GemmaClient 直接调用

```python
from src.tools.client import GemmaClient, LlamaServerManager
from src.skills.builtin import register_builtin_skills
from src.skills.registry import get_registry

register_builtin_skills()
server = LlamaServerManager()

try:
    server.start()
    client = GemmaClient(registry=get_registry(), server_manager=server)

    # 简单对话
    print(client.simple_chat("什么是量子计算？"))

    # 工具调用对话
    response = client.chat([
        {"role": "user", "content": "帮我计算 2^32"}
    ])
    print(response["choices"][0]["message"]["content"])

    # 动态连接 MCP fetch 服务器
    loaded = client.connect_mcp_stdio("fetch", ["uvx", "--from", "mcp-server-fetch", "mcp-server-fetch"])
    print(f"Loaded MCP tools: {loaded}")

    # 热加载 skill.md
    loaded = client.reload_skills()
    print(f"Reloaded skills: {loaded}")

finally:
    client.close()
    server.stop()
```

### cURL

```bash
# 简单对话
curl -s -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello!"}]}'

# 健康检查
curl -s http://127.0.0.1:8080/health

# 列出工具
curl -s http://127.0.0.1:8080/v1/tools

# 重新加载 skill.md
curl -s -X POST http://127.0.0.1:8080/v1/skills/reload

# 连接 MCP 服务器
curl -s -X POST http://127.0.0.1:8080/v1/mcp/connect \
  -H "Content-Type: application/json" \
  -d '{"name":"fetch","command":["uvx","--from","mcp-server-fetch","mcp-server-fetch"]}'
```

---

## 常见问题

### Q1: 模型加载失败 "Model file not found"

运行下载命令并验证：
```bash
python start.py download
python start.py info
```

### Q2: CUDA out of memory

- 减小上下文长度：修改 `src/config.py` 中 `N_CTX = 4096`
- 减少 GPU 层数：修改 `N_GPU_LAYERS = 20`（部分卸载到 CPU）
- 使用更小的量化版本（如 Q3_K_M）

### Q3: llama-server 启动失败

- 确认 GPU 驱动已更新到最新版本
- 确认 `llama-cpp-bin/` 目录下有 `ggml-cuda.dll` 和 `cudart64_12.dll`
- 尝试减少 GPU 层数：启动参数加 `-ngl 20`

### Q4: 端口冲突 [Errno 10048]

llama-server 和 FastAPI 不能使用同一端口。默认配置：
- FastAPI 代理：`8080`（对外）
- llama-server：`8081`（内部）

如需修改，编辑 `src/config.py` 中的 `SERVER_PORT` 和 `LLAMA_SERVER_PORT`。

### Q5: 工具调用不生效

4B 参数模型的工具调用能力有限，建议：
- 在提示中明确要求使用工具，如"请使用 calculator 工具计算..."
- 降低 `temperature` 至 0.3-0.7
- 增大 `max_tokens` 至 512+

### Q6: skill.md 解析失败

- YAML `description` 含冒号时**必须加引号**：`description: "text: with colon"`
- 确认代码块中有 `def execute(...)` 函数
- 查看启动日志中的 `[WARN]` 信息

### Q7: MCP 服务器连接失败

- 确认 MCP 服务器已正确启动并可访问
- stdio 方式：确认 `command` 中的可执行文件在 PATH 中
- HTTP 方式：确认 URL 和端口正确
- 查看 `[WARN]` 日志了解具体错误

### Q8: 推荐的推理参数

| 场景 | temperature | top_p | top_k |
|------|------------|-------|-------|
| 通用对话（官方推荐） | 1.0 | 0.95 | 64 |
| 工具调用 | 0.3-0.7 | 0.9 | 40 |
| 创意写作 | 1.0-1.2 | 0.95 | 64 |
| 精确问答 | 0.1-0.3 | 0.9 | 40 |

### Q9: 多模态（图片/音频）支持

模型原生支持多模态。使用 llama-cli：
```bash
llama-cpp-bin\llama-cli.exe -m models\*.gguf --mmproj models\mmproj-*.gguf --jinja -c 8192 -ngl 99
```
在聊天中输入 `/image <file>` 或 `/audio <file>` 添加多模态输入。
