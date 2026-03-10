---
name: interview-prep
description: Aegis-Isle 面试通关宝典 - 基于 Gabriella 个人背景和项目代码的面试准备系统
---

# 面试准备 Skill (Gabriella 特别版)

## 目标岗位画像
- **目标**: 2026年3月 广深 base 20k/月 AI 应用开发工程师
- **学历**: 广东工业大学计算机硕士 2023 毕业，gap 两三年
- **核心亮点**: Aegis-Isle 项目（RAG 多宇宙记忆系统 + SillyTavern 旁路挂载架构）

## 出题要求

### 题量与来源
- 30 道面试题
- 来源一：牛客网 2026 年 3 月广深 AI 应用开发面经
- 来源二：用户桌面 `C:\Users\MR\Desktop\forwork_面经` 收集的面经
- 来源三：必须阅读完项目代码后才能出题

### 每题格式（双回答模式）

每题包含：
1. **数据流举例**（500字）：从 SillyTavern 用户发消息开始，详细解释前端 → 后端 → Agent → 技术处理 的完整链路
2. **回答一 · Gabriella 的赛博茶话会**（200字）：
   - 用 RPG 游戏、K-pop 世界观、二次元梗把技术翻译成大白话
   - 附带代码坐标（文件路径 + 行号），方便复习
   - 帮助深度理解，内化成直觉
3. **回答二 · 专业面试官视角**（200字）：
   - 教 Gabriella 如何在面试官面前，基于自己对项目的理解专业地回答

### 架构重点

> [!IMPORTANT]
> 项目主线是 **SillyTavern 旁路记忆挂载架构**，每个回答都要基于项目代码。
> 传统的非 ST 架构也要提到一点作为对比。

### 核心技术模块（出题必须覆盖）

| 模块 | 代码路径 | 面试关键词 |
|------|----------|------------|
| RAG 多宇宙检索 | `src/aegis_isle/rag/st_memory_manager.py` | FAISS, 78 宇宙, 跨宇宙联合检索 |
| 向量嵌入引擎 | `src/aegis_isle/rag/embedder.py` | BAAI/bge-large-zh-v1.5, 1024维 |
| 知识图谱检索 | `src/aegis_isle/rag/graph_searcher.py` | 多层级记忆碎片 |
| 剧情回忆检索 | `src/aegis_isle/rag/episode_searcher.py` | Episode Plot 上帝视角 |
| 事件溯源总线 | `src/aegis_isle/rag/event_logger.py` | LifeEventBus, JSONL |
| 每日摘要聚合 | `src/aegis_isle/rag/daily_digest.py` | DailyDigest, ECoT, 深海总结 |
| 自治角色 Agent | `src/aegis_isle/agents/` | CharLifeAgent, AgentFetch |
| 状态管理系统 | `src/aegis_isle/core/state/` | Pydantic, 快照回滚, XML 指令提取 |
| OpenAI 兼容网关 | `src/aegis_isle/api/routers/openai_compat.py` | asyncio, 流式输出, 并发控制 |
| 记忆搜索路由 | `src/aegis_isle/api/routers/memory.py` | 四路并发 RAG 检索 |
| ST 前端扩展 | `st_extension/aegis-memory/index.js` | DOM Hook, CHAT_CHANGED 监听 |
| 面试系统集成 | `src/aegis_isle/interview/` | Love & Code, webhook 伪装挂载 |

### Gabriella 的个人调性

面试回答要贴合 Gabriella 的性格特征：
- **INFP**：外冷内热，思维敏锐，善于分析
- **直觉型洞察力**：能看透系统底层逻辑，绕过表层直击核心
- **技术宅模式**：配置 SillyTavern 后端时进入专注的"极客模式"
- **哲学底色**：结构自然主义、功能主义视角看待 AI 系统

## 操作步骤（给 Agent 的指令）

1. 读取 `C:\Users\MR\Desktop\forwork_面经` 目录下的面经文件
2. 扫描 `src/aegis_isle/` 下所有核心模块代码
3. 结合面经和代码，生成 30 题面试宝典
4. 每题按照"数据流举例 + 赛博茶话会回答 + 面试官视角回答"三段式输出
5. 将结果写入 `cowokers_ai/interview_guide_gabriella.md`

## 自动化联动

此 Skill 与 `nightly_pipeline.py` 联动：
- 当 nightly 管线检测到核心模块代码变更时
- 自动在 `interview_changelog.md` 中标记哪些面试题的代码引用可能需要更新
- 提示 Gabriella 复习对应的题目
