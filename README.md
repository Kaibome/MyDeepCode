# DeepCode

基于 **LangChain / LangGraph** 的多智能体「论文 → 理解与规划 → 代码实现」流水线，并通过 **MCP（stdio）** 接入搜索、文件系统、PDF 下载、文档切分、命令执行等工具。

## 功能概览

- **多阶段研究到代码**：从论文输入（本地路径 / URL）出发，完成分析、资源处理、工作区准备、（可选）文档分段、代码架构规划、引用与仓库相关阶段，以及迭代式代码实现与进度跟踪。
- **ReAct 工具调用智能体**：各阶段由带 MCP 工具的对话模型驱动。
- **MCP 集成**：通过 `langchain-mcp-adapters` 连接 `config.yaml` 中声明的 stdio 服务（如文件系统、搜索、PDF、分段脚本等）。
- **可配置 LLM**：支持 OpenAI / Anthropic / Google 等（由环境变量指定提供商与密钥）。

## 环境要求

- Python 3.10+（建议）
- 已安装 **Node.js**（`config.yaml` 中 `filesystem` 服务使用 `npx @modelcontextprotocol/server-filesystem`）
- 系统可用的 `python` 命令（用于启动各 MCP 子进程）

## 安装

1. 创建并激活虚拟环境（示例）：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. 在**仓库根目录**（包含 `deepcode` 包目录的那一层，例如 `MyDeepCode`）安装依赖：

```powershell
pip install -r deepcode/requirements.txt
```

## 环境变量（`.env`）

在 `deepcode` 目录下复制模板并编辑：

```powershell
copy deepcode\.env.example deepcode\.env
```

程序会从 **`deepcode/.env`** 加载变量（与 `multi_agent_research` 中 `load_dotenv` 路径一致）。

| 变量 | 是否必需 | 说明 |
|------|----------|------|
| `LLM_API_KEY` | **必需** | 对话模型 API 密钥（各提供商统一通过此变量传入给智能体） |
| `LLM_MODEL_PROVIDER` | 建议 | `openai` / `anthropic` / `google`，默认行为按代码中 `ReActAgent` 解析 |
| `LLM_MODEL_NAME` | 建议 | 模型名，如 `gpt-4o`、`claude-sonnet-4-20250514` 等 |
| `LLM_API_BASE` | 可选 | OpenAI 兼容接口的 Base URL（如官方或代理地址） |
| `PDF_DOWNLOADER_PATH` | 条件 | 指向 `pdf_downloader.py` 的路径；不设则不会为该智能体挂载 PDF 下载 MCP |
| `DOCUMENT_SEGMENTATION_PATH` | 条件 | 指向 `document_segmentation_server.py`；不设则不分段 MCP |
| `GITHUB_DOWNLOADER_PATH` | 可选 | 默认 `tools/git_command.py`（相对**当前工作目录**） |
| `CODE_IMPLEMENTATION_SERVER_PATH` | 条件 | 代码实现阶段 MCP，见 `code_implementation_flow_iterative` |
| `CODE_REFERENCE_INDEXER_PATH` | 条件 | 代码引用索引 MCP |

**从仓库根目录运行时**，工具脚本建议写成相对根目录的路径，例如：

```env
PDF_DOWNLOADER_PATH=deepcode/tools/pdf_downloader.py
DOCUMENT_SEGMENTATION_PATH=deepcode/tools/document_segmentation_server.py
```

（请按你本机仓库路径调整。）

**搜索（博查 Bocha）**：在 `config.yaml` 的 `mcp.servers.bocha-mcp.env` 中配置 `BOCHA_API_KEY`（见下文）；该键会与子进程环境合并。

> 说明：`utils/llm_utils.py` 中还提到 `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GOOGLE_API_KEY` 等，用于部分辅助逻辑；当前主流程智能体主要使用 **`LLM_API_KEY`**。

## 配置文件 `config.yaml`

程序在**当前工作目录**下查找 **`config.yaml`**（相对路径，非包内路径）。若在仓库根目录执行 `python -m deepcode`，请把 `deepcode/config.yaml` **复制或链接到根目录** 的 `config.yaml`，否则会退化为默认配置且 MCP 列表可能为空。

建议在 `config.yaml` 中按需修改：

- **`llm_provider`**：与所用密钥匹配的提供商偏好（`openai` / `anthropic` / `google`），与 `llm_utils` 等读取逻辑一致。
- **`default_search_server`**：默认搜索 MCP 服务名，需与 `mcp.servers` 下某一 key 一致（如 `bocha-mcp`）。
- **`openai` / `anthropic` / `google`**：各厂商默认模型名、token 上限策略等。
- **`document_segmentation`**：是否启用分段、`size_threshold_chars` 阈值。
- **`logging`**：日志级别、文件路径模式、是否输出到控制台等。
- **`mcp.servers`**：每个 stdio 服务的 `command`、`args`、`env`。  
  - **`filesystem`**：若 Windows 上 `npx` 不在 PATH，需自行保证可执行。  
  - **`bocha-mcp`**：将 `BOCHA_API_KEY` 改为真实密钥。  
  - 各 Python 服务的 `args` 多为 `tools/xxx.py`：子进程的工作目录通常与**你运行命令时的 cwd** 相关，若启动失败可改为**绝对路径**或保证 cwd 为 `deepcode` 且与 `PYTHONPATH` 一致。

## 运行方式

在**仓库根目录**（已安装依赖、已准备根目录下的 `config.yaml`、已配置 `deepcode/.env`）执行：

```powershell
python -m deepcode "file:///D:/path/to/paper.pdf"
```

或网络 PDF：

```powershell
python -m deepcode "https://example.com/paper.pdf"
```

运行产物（如 `deepcode_lab`）默认写在**当前工作目录**下。

## 测试与示例

- 多智能体快速试验：

```powershell
python -m deepcode.tests.try_multi_agent_research
```

- 单元测试：

```powershell
python -m unittest discover deepcode/tests
```

部分测试默认跳过或需要真实 API 与可用的 MCP 服务。

## 项目结构（节选）

```text
deepcode/
├── README.md
├── requirements.txt
├── .env.example
├── config.yaml          # 模板；运行时常拷贝到仓库根目录 config.yaml
├── main.py
├── __main__.py
├── agents/
├── agent_flow/
├── prompts/
├── tools/
├── utils/
└── tests/
```

## 说明

- 架构上对齐原有多智能体论文到代码的阶段划分；MCP 工具脚本位于 `tools/`，可按需单独用 stdio 调试。
- 若路径类错误频发，优先检查：**cwd**、`config.yaml` 是否在 cwd、`.env` 中工具脚本路径是否与 cwd 一致。
