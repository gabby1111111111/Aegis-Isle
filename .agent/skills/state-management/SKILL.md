---
name: state-management
description: Shujuku 状态管理系统的开发规范和操作指南
---

# Shujuku 状态管理技能

## 概述
Aegis-Isle 集成了 Shujuku 的结构化记忆系统。本技能描述了状态管理的架构、API 和开发规范。

## 核心架构

### 数据模型 (`src/aegis_isle/core/state/models.py`)
```
UserState
├── sheets: Dict[str, Sheet]    # 数据表集合
│   ├── sheet_global            # 全局状态 (地点/时间)
│   ├── sheet_hero              # 主角信息 (名字/职业)
│   ├── sheet_inventory         # 背包物品
│   └── sheet_quest             # 任务列表
├── version: int                # 状态版本号
└── user_id: str                # 用户标识
```

### 关键文件
| 文件 | 职责 |
|:---|:---|
| `models.py` | Pydantic 数据模型定义 |
| `manager.py` | 状态加载/保存/快照/回滚 |
| `extractor.py` | 从 LLM 输出提取 XML 指令 |
| `prompts.py` | Medusa Prompt 模板 |
| `context_injection.py` | 状态注入到聊天上下文 |

### 数据存储
- 状态文件: `data/state/{user_id}.json`
- 快照文件: `data/snapshots/{user_id}/{timestamp}.json`

## 状态管理 API

### 加载状态
```python
state_manager = StateManager()
user_state = await state_manager.load_state(user_id)  # async
```

### 保存状态
```python
await state_manager.save_state(user_id, user_state)  # async, 自动创建快照
```

### 状态转 Markdown (注入上下文)
```python
state_context = state_manager.get_context_string(user_state)
# 返回格式化的 Markdown 表格
```

### 注入到聊天消息
```python
from aegis_isle.core.state.context_injection import inject_state_context
messages = inject_state_context(messages, state_context)  # 同步!
```

> **重要**: `inject_state_context` 是**同步函数**，不要用 `await` 调用！

### 快照回滚
```python
snapshots = await state_manager.list_snapshots(user_id)
await state_manager.rollback_to_snapshot(user_id, snapshot_id)
```

## 开发规范

### 添加新的数据表
1. 在 `models.py` 中定义 `Sheet` 的 `sourceData` 和 `content` 结构
2. 在 `data/state/default.json` 中添加初始数据
3. 在 `prompts.py` 中更新 Medusa Prompt 以包含新表的操作规则
4. 测试: 确保 extractor 能正确解析新表的 XML 指令

### 状态更新的异步流程
```
SillyTavern 请求
    → chat_completions 端点
    → 加载用户状态
    → inject_state_context (同步)
    → 调用 LLM (流式)
    → 返回响应给 SillyTavern
    → BackgroundTask: 提取 XML → 更新状态 → 保存 JSON
```

### 注意事项
- `get_user_id_from_request(body)` 只接受 **一个参数**
- 状态文件使用 UTF-8 编码
- 每次保存自动创建快照 (用于回滚)
- 版本号 (`version`) 每次保存自动递增
