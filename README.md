# Gemma-4-E4B 本地部署指南

基于 llama.cpp 框架部署 Gemma-4-E4B-Uncensored-HauhauCS-Aggressive 模型，集成工具调用（Tool Calling）和技能系统（Skills）。

---

## 目录

1. [环境要求](#1-环境要求)
2. [快速开始](#2-快速开始)
3. [部署步骤](#3-部署步骤)
4. [架构说明](#4-架构说明)
5. [工具调用接口](#5-工具调用接口)
6. [Skills 技能系统](#6-skills-技能系统)
7. [API 接口文档](#7-api-接口文档)
8. [示例代码](#8-示例代码)
9. [常见问题](#9-常见问题)

---

## 1. 环境要求

| 项目 | 最低要求 | 推荐 |
|------|---------|------|
| GPU | NVIDIA GPU, 6GB+ VRAM | RTX 4070+ (8GB VRAM) |
| Python | 3.10+ | 3.12 |
| CUDA Driver | 12.1+ | 12.4+ |
| 磁盘空间 | 7 GB | 10 GB |
| 内存 | 8 GB | 16 GB |

> **注意**: 本项目使用 llama.cpp 官方预编译二进制文件，无需安装 CUDA Toolkit 或 C++ 编译器。

## 2. 快速开始

```bash
# 1. 安装 Python 依赖
pip install -r requirements.txt

# 2. 一键下载（模型 + llama.cpp 二进制）
python start.py download

# 3. 启动交互式聊天（含工具调用）
python chat.py

# 4. 或启动 API 服务器
python start.py serve --host 127.0.0.1 --port 8080
```

## 3. 部署步骤

### 3.1 安装依赖

```bash
pip install -r requirements.txt
```

依赖列表：
- `requests` — HTTP 请求
- `fastapi` + `uvicorn` — API 服务器
- `openai` — OpenAI 兼容客户端
- `pydantic` — 数据模型
- `huggingface_hub` — 模型下载

> 无需安装 llama-cpp-python 或编译任何 C++ 代码！

### 3.2 下载模型和二进制

```bash
python start.py download
```

将自动下载：
1. **llama.cpp 二进制** (~223 MB) — 含 `llama-server.exe`, `llama-cli.exe` 及 CUDA 运行时
2. **模型文件** (~5.0 GB) — `Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf`
3. **多模态投影** (~945 MB) — `mmproj-Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-f16.gguf`

> 下载大文件时建议设置 HF Token 以获得更快的下载速度：
> ```bash
> set HF_TOKEN=your_token_here
> python start.py download
> ```

### 3.3 验证部署

```bash
python start.py info
```

### 3.4 启动服务

**方式一：交互式聊天**
```bash
python chat.py
```

**方式二：OpenAI 兼容 API 服务器**
```bash
python start.py serve --host 127.0.0.1 --port 8080
```

**方式三：直接使用 llama-server**
```bash
llama-cpp-bin\llama-server.exe ^
  -m models\Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf ^
  --mmproj models\mmproj-Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-f16.gguf ^
  --host 127.0.0.1 --port 8080 ^
  -c 8192 --jinja -ngl 99
```

## 4. 架构说明

```
gemma4-e4b/
├── models/                          # 模型文件目录
│   ├── *.gguf                       # 主模型文件
│   └── mmproj-*.gguf                # 多模态投影文件
├── llama-cpp-bin/                   # llama.cpp 预编译二进制
│   ├── llama-server.exe             # OpenAI 兼容 API 服务器
│   ├── llama-cli.exe                # 命令行推理
│   ├── ggml-cuda.dll                # CUDA 后端
│   └── cudart64_12.dll              # CUDA 运行时
├── src/
│   ├── config.py                    # 全局配置
│   ├── server.py                    # FastAPI 工具调用代理服务器
│   ├── skills/                      # Skills 技能系统
│   │   ├── base.py                  # Skill 基类
│   │   ├── registry.py              # 技能注册中心
│   │   └── builtin/                 # 内置技能
│   │       ├── calculator.py        # 计算器
│   │       ├── file_ops.py          # 文件操作/日期时间/系统信息
│   │       └── code_exec.py         # 代码执行/Shell/网页抓取
│   └── tools/                       # 工具调用系统
│       ├── client.py                # LLM 客户端 + llama-server 管理
│       └── executor.py              # 工具执行引擎
├── chat.py                          # 交互式聊天入口
├── start.py                         # 统一启动入口
├── test.py                          # 测试脚本
└── requirements.txt                 # 依赖列表
```

### 核心流程

```
用户输入 → GemmaClient.chat() → llama-server API → LLM 推理
                                                        ↓
                                                模型返回 tool_calls?
                                               ╱Yes              ╲No
                                        ToolExecutor          返回文本响应
                                        执行对应 Skill              ↓
                                             ↓               输出给用户
                                        将结果追加到消息
                                             ↓
                                        再次调用 llama-server ←──┘
                                        (最多循环 10 次)
```

### 技术架构

- **推理后端**: llama.cpp b8839 (官方预编译 Windows CUDA 12.4 二进制)
- **API 协议**: OpenAI Chat Completions 兼容
- **工具调用**: OpenAI Function Calling 协议
- **技能系统**: Python 插件式架构，可动态注册

## 5. 工具调用接口

### 5.1 工具调用协议

本系统采用 OpenAI Function Calling 协议。模型在需要调用工具时，会在响应中生成 `tool_calls` 字段，包含工具名称和参数。系统自动执行工具并将结果返回给模型，模型根据结果生成最终回答。

### 5.2 内置工具列表

| 工具名 | 描述 | 参数 |
|--------|------|------|
| `calculator` | 数学表达式计算 | `expression`: 数学表达式 |
| `file_ops` | 文件读写/列表/检查 | `action`: read/write/list/exists, `path`: 路径, `content`: 写入内容 |
| `datetime` | 获取当前日期时间 | `action`: now/format, `format_str`: 格式字符串 |
| `system_info` | 系统信息查询 | `info_type`: basic/memory/disk |
| `code_exec` | 执行 Python 代码 | `code`: Python代码, `timeout`: 超时秒数 |
| `shell_exec` | 执行 Shell 命令 | `command`: 命令, `timeout`: 超时秒数 |
| `web_fetch` | 抓取网页内容 | `url`: URL, `max_length`: 最大长度 |

### 5.3 自定义工具

创建自定义工具只需继承 `Skill` 基类并注册：

```python
from src.skills.base import Skill
from src.skills.registry import get_registry

class MyCustomSkill(Skill):
    name = "my_tool"
    description = "我的自定义工具描述"
    parameters = {
        "type": "object",
        "properties": {
            "input": {
                "type": "string",
                "description": "输入参数",
            }
        },
        "required": ["input"],
    }

    def execute(self, input: str = "", **kwargs) -> str:
        return f"处理结果: {input}"

registry = get_registry()
registry.register(MyCustomSkill())
```

## 6. Skills 技能系统

### 6.1 Skill 基类

```python
class Skill(ABC):
    name: str           # 技能唯一标识
    description: str    # 技能描述（供模型理解）
    parameters: dict    # JSON Schema 格式的参数定义

    def execute(self, **kwargs) -> str:
        """执行技能，返回字符串结果"""
        raise NotImplementedError

    def to_openai_tool(self) -> dict:
        """转换为 OpenAI Tool 格式"""
```

### 6.2 SkillRegistry 注册中心

```python
registry = get_registry()

registry.register(my_skill)          # 注册
registry.unregister("skill_name")    # 注销
registry.list_skills()               # 列出所有
registry.get_openai_tools()          # 获取 OpenAI Tools 格式
registry.execute("calculator", expression="2+2")  # 执行
```

### 6.3 技能加载流程

系统启动时，`register_builtin_skills()` 自动注册所有内置技能。如需加载自定义技能：

```python
from src.skills.builtin import register_builtin_skills
from src.skills.registry import get_registry

register_builtin_skills()
registry = get_registry()
registry.register(MyCustomSkill())
```

## 7. API 接口文档

启动 API 服务器后，提供以下 OpenAI 兼容接口：

### 7.1 Chat Completions

```
POST /v1/chat/completions
```

**请求体：**
```json
{
  "model": "gemma-4-e4b",
  "messages": [
    {"role": "user", "content": "计算 123 * 456"}
  ],
  "tools": [...],
  "temperature": 1.0,
  "top_p": 0.95,
  "top_k": 64,
  "max_tokens": 2048,
  "stream": false
}
```

### 7.2 其他接口

| 接口 | 方法 | 描述 |
|------|------|------|
| `/v1/models` | GET | 列出可用模型 |
| `/v1/models/{id}` | GET | 获取模型信息 |
| `/v1/tools` | GET | 列出可用工具 |
| `/health` | GET | 健康检查 |

## 8. 示例代码

### 8.1 Python 客户端调用

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="gemma-4-e4b",
    messages=[{"role": "user", "content": "你好！"}],
)
print(response.choices[0].message.content)
```

### 8.2 直接使用 GemmaClient（含工具调用）

```python
from src.tools.client import GemmaClient, LlamaServerManager
from src.skills.builtin import register_builtin_skills
from src.skills.registry import get_registry

register_builtin_skills()
server = LlamaServerManager()
server.start()

client = GemmaClient(registry=get_registry(), server_manager=server)

# 简单对话
answer = client.simple_chat("什么是量子计算？")
print(answer)

# 带工具的对话（自动调用工具）
response = client.chat([
    {"role": "user", "content": "帮我计算 2^32"}
])
print(response["choices"][0]["message"]["content"])

server.stop()
```

### 8.3 cURL 调用

```bash
curl -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma-4-e4b","messages":[{"role":"user","content":"Hello!"}]}'
```

### 8.4 自定义 Skill 示例

```python
from src.skills.base import Skill
from src.skills.registry import get_registry
from src.tools.client import GemmaClient, LlamaServerManager
from src.skills.builtin import register_builtin_skills

class TranslatorSkill(Skill):
    name = "translator"
    description = "Translate text between languages"
    parameters = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Text to translate"},
            "target_lang": {"type": "string", "description": "Target language"}
        },
        "required": ["text", "target_lang"]
    }

    def execute(self, text: str = "", target_lang: str = "en", **kwargs) -> str:
        return f"[Translation to {target_lang}]: {text}"

register_builtin_skills()
registry = get_registry()
registry.register(TranslatorSkill())

server = LlamaServerManager()
server.start()
client = GemmaClient(registry=registry, server_manager=server)

response = client.chat([
    {"role": "user", "content": "Translate 'Hello World' to Chinese"}
])
print(response["choices"][0]["message"]["content"])
server.stop()
```

## 9. 常见问题

### Q1: 模型加载失败 "Model file not found"

**解决**:
```bash
python start.py download
python start.py info
```

### Q2: CUDA out of memory

**解决**:
- 减小上下文长度：修改 `src/config.py` 中 `N_CTX = 4096`
- 使用更小的量化版本（如 Q3_K_M）

### Q3: llama-server 启动失败

**解决**:
- 确认 GPU 驱动已更新到最新版本
- 确认 `llama-cpp-bin/` 目录下有 `ggml-cuda.dll` 和 `cudart64_12.dll`
- 尝试减少 GPU 层数：启动参数加 `-ngl 20`

### Q4: 工具调用不生效

**原因**: 4B 参数模型的工具调用能力有限。

**解决**:
- 在用户消息中明确要求使用工具，如"请使用 calculator 工具计算..."
- 调整 `temperature` 为 0.3-0.7 以减少随机性
- 增大 `max_tokens` 参数

### Q5: httpx 请求返回 502

**解决**: 本项目使用 `requests` 库替代 `httpx`。如遇到 502 错误，请确认：
- llama-server 正在运行
- 端口未被占用
- 使用 `requests` 库而非 `httpx`

### Q6: 推荐的推理参数

根据 Google Gemma 4 官方建议：
- `temperature = 1.0`, `top_p = 0.95`, `top_k = 64`

对于工具调用场景，建议：
- `temperature = 0.3`, `top_p = 0.9`, `top_k = 40`

### Q7: 多模态（图片/音频）支持

模型原生支持多模态。使用 llama-cli：
```bash
llama-cpp-bin\llama-cli.exe -m models\*.gguf --mmproj models\mmproj-*.gguf --jinja -c 8192 -ngl 99
```
在聊天中输入 `/image <file>` 或 `/audio <file>` 添加多模态输入。
