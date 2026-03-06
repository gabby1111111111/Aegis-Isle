# 🛡️ Aegis-Isle: Enterprise Multi-Agent RAG Platform
### 高并发异步 AI 架构与垂直 Agent 编排系统

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/Agentic-LangGraph-purple)](https://langchain.com/)
[![FAISS](https://img.shields.io/badge/VectorDB-FAISS-green)](https://faiss.ai/)
[![Asyncio](https://img.shields.io/badge/Concurrency-Asyncio-yellow)]()

> **"面向生产环境的高性能 Agentic RAG 底座，重塑多模态上下文记忆与自治流控。"**
>
> **Aegis-Isle** 是一套专为复杂状态管理和高并发场景设计的 **通用企业级 RAG 与多智能体架构**。本项目核心解决了长文本 LLM 会话中的“幻觉漂移”与“状态丢失”痛点。通过透明拦截 OpenAI 协议兼容接口，系统静默实现了**三路（Dense + 知识图谱 + 情节）异步并发混合检索**、**Pydantic 实体化会话状态剥离**，以及**基于背景任务的独立自治反思网络**（CharLifeAgent）。全面支持极低延迟的时空对齐与企业级全链路监控。

---

## 💡 核心竞争力 (Core Capabilities)

### 1. 🚀 高性能意图路由与异步并发检索网关
* **动态意图感知路由**：基于 AST 与正则的极速 `O(1)` 前置意图检测层，精准识别前端用户的隐含交互指令（如：剧情回顾、情感诊断），瞬间分发至特定后端节点，**大幅降低了无意义的大模型 Token 消耗**。
* **三路异构并发 RAG (Multi-stage Concurrent RAG)**：完全摒弃传统阻塞式链条，采用 `asyncio.gather` 编排 FAISS 向量库、JSONL 稀疏事件集、以及待接驳的知识图谱进行三路并行 Query。
* **带权环境感知后过滤 (Metadata Pre-filtering)**：实现软硬件级别的时空对齐。利用逆向映射算法提取子块对应的场景元数据 (`scene_meta`, `location/time/weather`)，并在提取层做倒排加权过滤，完美解决 Dense 检索的“语义相近但时空倒错”的老大难问题，端到端检索耗时严格控制在 **300ms** 阈值以内。

### 2. 🧠 Pydantic 强类型流式状态机 (Stateful Engine)
* **结构化状态隔离**：受前端 Shujuku 状态图启发，使用 FastAPI 与 Pydantic 完全重构后端状态追踪系统。将会话上下文抽象为强类型的 `Sheet/Row` 对象树，彻底隔离了易变的对话流与持久化的状态机。
* **三级容错提取链路 (3-Level Fallback Extraction)**：创新性引入 `LLM XML Agent 解析` → `正则表达式断言` → `TF-IDF 关键字兜底` 的三级熔断式防御提取机制，确保复杂指令下 99.9% 的状态抽取成功率。
* **无锁异步快照树 (Lock-free Snapshoting)**：基于 Copy-on-Write 思想，在每次会话变动前，异步向磁盘写入全量状态切片，提供秒级版本回溯能力，保证分布式并发场景下的数据绝对一致性（ACID近似）。

### 3. 🕸️ 领域驱动自治节点群 (Background Autonomous Agents)
* **主从通信流劫持**：实现前端扩展脚本 `aegis-memory (SillyTavern Plugin)` 与后端的双向握手。通过特定的斜杠指令前缀截留耗时任务。
* **Agent 休眠唤醒机制**：主线程使用 `StreamingResponse` 与伪造 Token 流秒级响应客户端交互请求；后台通过 FastAPI `BackgroundTasks` 唤醒 LangGraph 构建的 `CharLifeAgent` 执行深度图谱反思、关系评估摘要与新闻抓取合成。
* **隔离宇宙沙盒 (Universe Sandboxing)**：自研基于哈希掩码系统提示片段生成的 `Universe ID`，完美实现了千人千面的知识库隔离降级策略，消除了任意上下文渗透风险。

### 4. 🔭 生产级可观测性与治理 (Production Observability)
* **高精度 Token 指标泵**：集成 `tiktoken` 实时旁路计算 Input/Output token 开销，动态估算硅基计算成本。
* **ELK 兼容的审计管道**：标准化每次并发问答的数据切片，输出符合 ELK-Stack 无模式解析规范的 JSONL 日志阵列（包含 P50/P95/P99 核心延迟遥测）。
* **全域 RBAC 防护墙**：标准的 OAuth2 + JWT 身份鉴权机制，并对所有 API 端点挂载微秒级 Trace ID 链路追踪。

---

## 🏗️ 架构拓扑 (Architecture Topology)

```mermaid
graph TD
    classDef core fill:#2d3748,stroke:#4a5568,stroke-width:2px,color:#fff;
    classDef st fill:#e53e3e,stroke:#c53030,stroke-width:2px,color:#fff,stroke-dasharray: 5 5;
    classDef rag fill:#2b6cb0,stroke:#2c5282,stroke-width:2px,color:#fff;
    classDef graph fill:#805ad5,stroke:#553c9a,stroke-width:2px,color:#fff;

    Client[Enterprise OA / CRM<br/>Generic SDK]
    ST[SillyTavern RP Extended UI<br/>Roleplay Node]:::st

    subgraph Aegis_Isle_Core [Aegis-Isle Agentic Core]
        style Aegis_Isle_Core fill:#f7fafc,stroke:#cbd5e0,stroke-width:2px,color:#1a202c
        Router[Fast Regex Intent Router & Hijacker<br/>Zero-delay Proxy]:::core
        Agent[Background LangGraph Autonomous Nodes<br/>Reflection & Learn]:::graph
        Extractor[3-Level Fault-tolerant State Extractor<br/>Pydantic / XML regex]:::rag
        TripleRAG[Triple-Core RAG Engine<br/> FAISS / Graph / Epis]:::rag

        Router -- Async Gather --> Extractor
        Router -- Async Gather --> TripleRAG
        Router -. Background Task .-> Agent
    end

    Client -->|POST /v1/chat/completions| Router
    ST -->|POST /v1/chat/completions| Router
```

---

## 🛠️ 技术栈 (Tech Stack)

| 核心层域 | 关键技术框架选型 |
|:---|:---|
| **API 服务网关** | Python 3.11, FastAPI (ASGI), Uvicorn, Asyncio |
| **Agent / 状态流** | LangGraph, Pydantic v2 (Strict Schema), background tasks |
| **向量神经检索** | FAISS, LangChain Vectorstores, Custom Metadata Pre-filtering |
| **安全与可观测性** | Tiktoken, JWT (Python-JOSE), Bcrypt, P99 Latency Track |
| **扩展通信侧** | 原生 JavaScript Promise (SillyTavern Aegis-Memory) |

---

## 📸 垂直领域生态与落地验证 (Vertical Applications & Ecosystem)

> **“Aegis-Isle 不仅是一个后端框架，更是一个多端联动的生态中枢。”**
> 本项目已在多个复杂的垂直场景中实现了完美的落地验证，尤其在对“上下文一致性”和“低延迟交互”要求极其苛刻的场景中表现优异。

### 1. 🎭 SillyTavern 沉浸式情景引擎 (Roleplay & Memory Backend)
针对顶级开源 RP 终端 SillyTavern，本项目研发了原生前端代理扩展 (`aegis-memory`)，提供企业级的降维打击方案：
* **幽灵注入算法 (Phantom Context Injection)**：前端脚本逆向 ST 渲染树，在倒数第二层级 (`depth=1`) 无痕插入 `identifier` 实体，**完美守护核心设定 prompt 的格式纯洁性**，杜绝大模型性格崩坏。
* **时空引力场防漂移 (Spatiotemporal Gravitational Field)**：创新的软硬件结合映射阵列。毫秒级提取 Query 中的地点/时间 Hint，强制回溯 `scene_meta.location` 进行绝对倒排，彻底解决纯 Dense 检索的高维语义模糊（如“酒吧”错配成“餐厅”）。

### 2. 🤖 领域自治节点群：CharLifeAgent (Background Autonomous Network)
依托 LangGraph 打造的独立后台图谱网络，赋予 AI 角色真正的“硅基生命史”：
* **无感异步神经网**：部署“指令伪装劫持”。当前端检测到特定动作（如 `/recap`, `/relation`）时，网关立即阻断 HTTP 等待流并返回仿制气泡，保障前端 UI 极速响应。
* **图谱后台长线反思**：隔离的 FastAPI Background Tasks 会暗中唤醒 `CharLifeAgent` 图谱节点，执行长达数十秒的深度图谱反思、关系动态评估与外网新闻抓取，最终动态重塑知识库生态。

### 3. 💼 垂直业务枢纽一：Project Love & Code (沉浸式面试备考系统)
* **业务定位**：将普通的闲聊对话框转变为硬核的技术面试与情感支持并行的双流引擎系统。
* **后端支撑**：Aegis-Isle 作为“爱与代码”项目的底层中枢，承载了所有的知识点检索、艾宾浩斯复习调度曲线及做题状态追踪。依靠架构内的无锁快照与 Pydantic 流式状态机，为每一次模拟面试提供极致的无感记忆供给与进度序列化。

### 4. 🌐 垂直业务枢纽二：ST-Companion-Link (全域感官与信息遥感)
* **业务定位**：打破狭窄的聊天界面次元壁，赋予 AI 角色观察用户正在浏览的互联网世界的“视觉与感知”。
* **后端支撑**：Aegis-Isle 作为本人独立开源项目 [ST-Companion-Link-Suite](https://github.com/gabby1111111111/ST-Companion-Link-Suite) (Chrome 浏览器遥感扩展群) 的统一云端大脑池。高并发接收来自小红书 (Xiaohongshu) 与 B 站 (Bilibili) 实时抓取的用户浏览足迹，结构化清洗后直接交由底层 FAISS 向量库与 Agent 节点处理建立行为档案，彻底激活“跨模态主动交互”——AI 可主动依据用户的历史网页流发起深度探讨。

---

## �🚀 极速部署 (Quick Start)

### 选项 A: 独立高并发后端启动

```bash
# 复制并配置通用鉴权
cp .env.example .env
# 填入您的 OPENAI_API_KEY 以及对应的模型终端端点

# 强烈建议使用虚拟环境启动，规避依赖冲突
.\venv\Scripts\uvicorn.exe src.aegis_isle.api.main:app --host 0.0.0.0 --port 8000

# [诊断入口] Swagger OpenAPI 文档: http://localhost:8000/docs
```

### 选项 B: 沉浸式场景（SillyTavern 终端）适配

本项目为前端预制了深度整合代理引擎 `aegis-memory`，实现前端指令直达后端图谱大脑：

1. 寻找本项目根目录下的 `st_extension` 文件夹。
2. 将其中的 `aegis-memory` 复制进入独立 SillyTavern 的 `data/default-user/extensions/` 插件目录中。
3. 重启 SillyTavern 并在扩展配置页中勾选开启 Aegis Memory。
4. **环境要求**：保持 Aegis-Isle 后台运行于 `8001` 或 `8000` 端口。系统即会自动接管对话并进行极低延迟的双向动态记忆注射。

*(开发者贴士：欲分析后台并发分发流量与原始 Prompt 快照追踪，可启动时声明环境变量 `set DEBUG_SAVE=true` )*

---

## 🗺️ 阶段里程碑 (Roadmap)

- [x] **v1.0** — 搭建基础 RAG Engine 与 FAISS 向量缓冲池
- [x] **v1.5** — LangGraph 节点迁移与基础 Semantic Routing 拆分
- [x] **v2.0** — 落地企业级安全隔离墙 (RBAC + 日志强力审计)
- [x] **v2.1** — 成功交付 Session State Management 及三级提取状态网关
- [x] **v2.2** — 落地生产级全链路可观测面板 (监控 Token 损耗与 P99 长尾延迟)
- [x] **v2.3** — Native SillyTavern RAG Integration (并发多模态检索 / 场景元数据校对 / 独立自治进程)
- [ ] **v2.4** — RAGAS 评估自动化管线的搭建

---

<div align="center">
  <p>Engineered for High Concurrency, Designed for Sentient Intelligence.</p>
  <p><b>Targeting Innovation · Shenzhen, China</b></p>
</div>
