# 电商智能客服 Agent（RAG + LangGraph）

一个面向电商售后场景的 **RAG 知识库问答 + AI Agent** 项目：既能基于知识库做带引用溯源的问答，也能让大模型自主决策调用工具（查商品、查订单），并支持多轮对话记忆。

> 从零手写 RAG 全链路（切片 → 向量化 → 检索 → 生成），再基于 LangGraph 升级为可自主决策的 Agent。

## ✨ 功能特性

### 1. RAG 知识库问答（`POST /qa`）
- 电商售后知识库文档切片 → Embedding 向量化 → Qdrant 余弦相似度检索
- 检索结果去重后拼入 Prompt，由大模型生成答案
- 答案**带引用溯源**（标题 + 原文链接 + 相似度分数）

### 2. RAG + Agent（`POST /agent`）
- 基于 **LangGraph** 构建 `LLM ↔ 工具` 循环图：LLM 自主决策 → 调工具 → 看结果 → 再决策 → 回答
- 集成三类工具，LLM 根据用户意图自动选择：
  | 工具 | 用途 |
  |---|---|
  | `retrieve` | 查询售后知识库（退货政策、价格保护、保修等规则类问题） |
  | `search_products` | 按关键词搜索商品（推荐、比价类问题） |
  | `get_order` | 按订单号查询订单状态（物流、订单类问题） |
- **多轮对话记忆**：按 `session_id` 维护会话历史，支持上下文追问

### 3. 可视化聊天界面
- HTML 单页聊天前端，输入问题 → 实时回答，多轮连续对话

## 🧱 技术栈

| 层 | 技术 |
|---|---|
| 后端框架 | FastAPI + Uvicorn |
| Agent 编排 | LangGraph（StateGraph + 条件边循环） |
| 大模型 | 阿里云百炼 `qwen-turbo`（OpenAI 兼容接口） |
| 向量化 | 百炼 `text-embedding-v3`（1024 维） |
| 向量数据库 | Qdrant（本地文件模式） |
| 数据处理 | pandas |
| 配置管理 | pydantic-settings + .env |
| 前端 | 原生 HTML / JavaScript（fetch 调用 API） |

## 📁 项目结构

```
Agent_project/
├── app/                     # 应用层（FastAPI + Agent）
│   ├── main.py              # FastAPI 入口，托管前端页面
│   ├── config.py            # 配置（pydantic-settings 读取 .env）
│   ├── schemas.py           # 请求/响应 Pydantic 模型
│   ├── routes.py            # API 路由（/qa、/agent）
│   ├── services.py          # RAG 问答编排（检索→拼Prompt→生成→引用）
│   ├── tools.py             # 工具函数（查商品、查订单）
│   ├── agent.py             # Function Calling 示例（单轮工具调用）
│   └── graph.py             # LangGraph 编排（LLM↔工具循环 + 条件边）
├── ingestion/               # 数据摄取层
│   ├── chunker.py           # 文档切片（≤400字）
│   ├── embedder.py          # 向量化（百炼 Embedding）
│   └── indexer.py           # 建索引（写入 Qdrant）
├── search/                  # 检索层
│   └── retriever.py         # 语义检索（向量召回 + doc_id 去重）
├── static/
│   └── index.html           # 聊天前端页面
├── data/
│   ├── raw/                 # 知识库源文档（20 篇售后规则）
│   └── mock/                # 模拟商品 / 订单数据 + 生成脚本
├── tests/                   # 测试目录
├── requirements.txt         # 依赖清单
└── .env.example             # 配置模板（复制为 .env 填写）
```

## 🚀 快速开始

### 环境要求
- Python 3.10+
- 一个 OpenAI 兼容接口的 API Key（本项目使用阿里云百炼）

### 1. 克隆与安装

```bash
git clone <your-repo-url>
cd Agent_project

# 创建虚拟环境（可选但推荐）
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. 配置密钥

```bash
cp .env.example .env
```

编辑 `.env`，填入你的密钥：

```ini
# 模型服务（OpenAI 兼容接口，示例为阿里云百炼）
LLM_API_KEY=sk-xxxxx
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen-turbo
EMBED_MODEL=text-embedding-v3
```

> ⚠️ **安全提醒**：`.env` 已加入 `.gitignore`，切勿提交到 GitHub。

### 3. 建立知识库索引

首次运行需要把知识文档切片、向量化并写入 Qdrant：

```bash
python -m ingestion.indexer
```

看到 `入库完成：42 个向量点` 即成功。

### 4. 启动服务

```bash
python -m uvicorn app.main:app --reload
```

打开浏览器访问：

| 地址 | 说明 |
|---|---|
| `http://127.0.0.1:8000/` | 聊天界面（前端演示） |
| `http://127.0.0.1:8000/docs` | Swagger 交互式 API 文档 |

### 5. 验证

```bash
# 单独测试工具函数
python -m app.tools

# 单独测试 Function Calling（项目②单轮版）
python -m app.agent

# 单独测试 LangGraph 编排（项目②循环版）
python -m app.graph
```

## 📡 API 说明

### `POST /qa` — RAG 知识库问答

**请求：**
```json
{
  "question": "7天无理由退货能退什么？"
}
```

**响应：**
```json
{
  "answer": "7天无理由退货适用于大部分商品，但以下商品不适用：……",
  "references": [
    {
      "title": "7天无理由退货规则",
      "source_url": "https://...",
      "score": 0.85
    }
  ]
}
```

### `POST /agent` — Agent 对话（多轮）

**请求：**
```json
{
  "question": "O20260918001 这个订单到哪了？",
  "session_id": "user_abc123"
}
```

**响应：**
```json
{
  "answer": "您的订单 O20260918001 已支付，订单详情如下：……"
}
```

> `session_id` 相同即视为同一会话，可进行多轮上下文追问。

## 💡 演示示例

| 用户输入 | Agent 行为 |
|---|---|
| 7天无理由退货能退什么？ | 调用 `retrieve` 查知识库，带引用回答 |
| 帮我推荐一款手机 | 调用 `search_products` 返回商品列表 |
| O20260918001 这个订单到哪了？ | 调用 `get_order` 查询订单状态 |
| 那它多少钱？（追问上一轮商品） | 结合会话记忆理解"它"指代，继续回答 |

## 🗺️ 项目演进（从 Demo 到工程化）

当前项目已完成核心功能链路，属"个人 Demo 偏上"水平。后续可补充的工程化方向：

- [ ] **接口鉴权**：API Key / JWT 保护 `/agent` 接口
- [ ] **错误重试**：LLM 调用失败自动重试与超时处理
- [ ] **任务级评测集**：构建 golden 问题集，量化回答正确率与工具调用准确率
- [ ] **工具护栏**：敏感操作二次确认，防 Prompt Injection
- [ ] **对话持久化**：会话历史从内存迁移到 Redis / SQLite
- [ ] **成本控制**：Token 用量统计与预算限制
- [ ] **流式输出**：SSE 打字机效果

## 📄 License

MIT
