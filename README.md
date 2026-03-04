
# 🛡️ Aegis-Isle: Enterprise Multi-Agent RAG Platform
### General-Purpose Knowledge Governance & Vertical Agent Orchestration

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/Agentic-LangGraph-purple)](https://langchain.com/)
[![FAISS](https://img.shields.io/badge/VectorDB-FAISS-green)](https://faiss.ai/)
[![Security](https://img.shields.io/badge/OAuth2-RBAC-red)]()
[![Observability](https://img.shields.io/badge/Observability-ELK%20Compatible-orange)]()

> **"A general-purpose, production-ready RAG backend — with stateful agent memory and real-time observability."**
>
> Aegis-Isle 是一个**通用企业级 RAG 平台**。它兼容任意 OpenAI 协议客户端，提供混合检索、会话状态管理、OAuth2/RBAC 安全治理和 ELK 兼容审计日志。SillyTavern RP 集成作为其**垂直领域扩展**之一展示了平台的适应性。

---

## 🏗️ Architecture Overview (通用架构)

```
┌─────────────────────────────────────────────────────────┐
│              Aegis-Isle Core Platform                    │
│                                                          │
│  ┌──────────────┐  ┌─────────────┐  ┌────────────────┐  │
│  │ Context      │  │ Token       │  │ ELK Audit      │  │
│  │ Injection    │  │ Metrics     │  │ Logging        │  │
│  │ Middleware   │  │ P50/P95/P99 │  │ log_llm_call() │  │
│  └──────────────┘  └─────────────┘  └────────────────┘  │
│  ┌──────────────────────────────────────────────────┐    │
│  │     Session State Management                     │    │
│  │     (inspired by AlbusKen/shujuku)               │    │
│  │     Pydantic Model · 3-Level Extractor · Snapshot│    │
│  └──────────────────────────────────────────────────┘    │
│  FAISS · Hybrid Retrieval · JWT/RBAC · LangGraph        │
└──────────────────────┬──────────────────────────────────┘
                       │  POST /v1/chat/completions
          ┌────────────┴────────────┐
          ▼                         ▼
  ┌──────────────┐         ┌─────────────────────┐
  │  Any OpenAI- │         │  SillyTavern RP      │
  │  Compatible  │         │  (Vertical Extension)│
  │  Client/App  │         │  角色扮演 + 状态记忆  │
  └──────────────┘         └─────────────────────┘
```

**所有核心功能（状态管理、Token 统计、审计日志、快照回滚）均通过标准 OpenAI 兼容接口暴露，适用于任意应用场景。**

---

## 🚀 Core Technical Highlights

### 1. 🧠 Session State Management (会话状态管理)

> 受 [AlbusKen/shujuku](https://gcore.jsdelivr.net/gh/AlbusKen/shujuku@mov1.1/index.js) 启发，移植并扩展为 Python/FastAPI 的结构化状态引擎。

- **Structured State Model**: Pydantic 强类型 `Sheet/Row` 模型管理任意结构化会话状态（库存、任务、用户属性等）
- **Three-Level Extraction**: LLM XML 解析 → 正则表达式 → 关键词匹配的三级容错提取链
- **Context Injection Middleware**: 状态序列化为 Markdown，注入任意 LLM System Prompt
- **Atomic Snapshot & Rollback**: 每次变更前自动创建 JSON 快照，支持版本回滚
- **Async Background Update**: 状态更新完全异步，不阻塞主响应流，overhead < 50ms

**通用应用场景:**
| 行业 | 状态内容 |
|:---|:---|
| RP/游戏 | 角色背包、任务、属性 |
| 电商客服 | 购物车、订单历史、用户偏好 |
| 医疗问诊 | 症状记录、用药历史、诊断进度 |
| 在线教育 | 学习进度、掌握度、错题集 |

### 2. 🔭 Production Observability (生产级可观测性)

- **Token Metrics**: `tiktoken` 实时统计 prompt/completion token，SiliconFlow/OpenAI 定价估算
- **P50/P95/P99 Latency**: 按端点分组延迟分位数，>5s 慢请求自动 WARNING
- **ELK-Compatible Audit Log**: 每次 LLM 调用写入 JSONL 审计日志（model/tokens/latency/cost/user_id）
- **Dashboard API**: `/api/v1/metrics/dashboard` 实时面板，CSV 导出

### 3. 🔐 Enterprise Governance & Security

- **Fine-Grained RBAC**: 三层角色权限体系 (User/Admin/SuperAdmin)
- **OAuth2 + JWT**: 标准认证流程，Bcrypt 密码加密，请求级 Trace ID
- **Structured Audit Log**: ELK-Stack 兼容，追踪权限变更与敏感操作

### 4. 🧩 Hybrid RAG Engine

- **Multi-Stage Retrieval**: Dense Vector（语义）+ BM25（关键词）混合检索
- **FAISS Vector Engine**: 相关度阈值 > 0.34 精准过滤
- **Cross-Encoder Re-ranking**: 降低 "Lost in the Middle" 现象
- **Dynamic Ingestion**: 支持 PDF/MD/Image 实时解析与向量化

---

## 🛠️ Tech Stack

| 层次 | 技术 |
|:---|:---|
| **Core Framework** | Python 3.11, FastAPI, Pydantic v2, Async/Await |
| **AI Orchestration** | LangGraph, LangChain, OpenAI SDK |
| **State Management** | Inspired by [AlbusKen/shujuku](https://github.com/AlbusKen/shujuku), Python reimplementation |
| **Vector Database** | FAISS, Qdrant |
| **Observability** | tiktoken, Loguru, ELK-compatible JSONL, P50/P95/P99 |
| **Security** | Python-JOSE (JWT), Passlib (Bcrypt), RBAC |
| **Deployment** | Docker Compose, Uvicorn |

---

## 📸 Vertical Domain Application: Project Love & Code

> Aegis-Isle 通用底座的一个垂直落地案例——将 RP 接口适配为沉浸式面试备考系统。  
> SillyTavern 作为前端，Aegis-Isle 提供有状态 RAG 后端。

- **Features**: 角色扮演式知识问答、艾宾浩斯复习调度、学习进度状态追踪
- **Architecture**: 通用状态管理 + RAG 检索 → SillyTavern 角色卡适配

![UI](./pre/interview_ui_v0.jpg)

---

## 🚀 Quick Start

### Option A: 轻量化服务器（推荐快速体验）

```bash
cp .env.example .env
# 填入 OPENAI_API_KEY

.\venv\Scripts\uvicorn.exe test_server:app --host 0.0.0.0 --port 8001

# OpenAI 兼容接口:  http://127.0.0.1:8001/v1/chat/completions
# Token 统计面板:   http://127.0.0.1:8001/api/v1/metrics/dashboard
# 状态查看:        http://127.0.0.1:8001/v1/state/{user_id}
```

### Option B: 企业级全功能服务器

```bash
.\venv\Scripts\uvicorn.exe src.aegis_isle.api.main:app --host 0.0.0.0 --port 8000
# Swagger UI: http://localhost:8000/docs
```

### Option C: Docker

```bash
cp .env.example .env && docker-compose up -d --build
```

---

## 📡 API Reference

| Endpoint | 描述 |
|:---|:---|
| `POST /v1/chat/completions` | OpenAI 兼容接口（流式 + 状态管理） |
| `GET /v1/state/{user_id}` | 查看用户当前状态 |
| `GET /v1/state/{user_id}/snapshots` | 列出历史快照 |
| `POST /v1/state/{user_id}/rollback` | 回滚到指定快照 |
| `GET /api/v1/metrics/dashboard` | Token 统计 + 延迟面板 |
| `GET /api/v1/metrics/export` | 导出 CSV 报告 |
| `POST /api/v1/auth/token` | 获取 JWT Bearer Token |
| `POST /api/v1/documents/upload` | 上传文档到向量库 |
| `GET /api/v1/health` | 健康检查 |

---

## 🗺️ Roadmap

- [x] **v1.0** — RAG Engine + FAISS Vector Store
- [x] **v1.5** — LangGraph Migration + Semantic Routing
- [x] **v2.0** — Enterprise Security (RBAC + Audit Logs)
- [x] **v2.1** — Session State Management + OpenAI-Compatible Stateful API
- [x] **v2.2** — Production Observability (Token Metrics + P99 Latency + ELK Audit)
- [ ] **v2.3** — RAGAS Evaluation Pipeline

---

## � Acknowledgements

- Session state management system inspired by [AlbusKen/shujuku](https://github.com/AlbusKen/shujuku) (JavaScript implementation for SillyTavern). Python reimplementation with Pydantic, FastAPI async patterns, and atomic snapshot system by this project.

---

<div align="center">
  <p>Engineered for Scalability, Designed for Intelligence.</p>
</div>
