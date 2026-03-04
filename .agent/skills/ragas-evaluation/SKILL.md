---
name: ragas-evaluation
description: 基于 SillyTavern 真实 RP 对话运行 RAGAS 评估，生成 Faithfulness/Recall/Precision 报告
---

# RAGAS 评估技能

## 概述
本技能用于对 Aegis-Isle RAG 系统进行质量评估。评估数据来源于 SillyTavern 的真实角色扮演对话，而非人工编造的测试集。

## 前置条件
- `pip install ragas datasets` 已安装
- Aegis-Isle 服务器正在运行 (`uvicorn test_server:app --port 8001`)
- SillyTavern 已连接到 `http://localhost:8001/v1`
- SiliconFlow API Key 已配置在 `.env`

## 评估流程

### Step 1: 采集对话数据
在 `test_server.py` 的 `chat_completions` 端点中，ConversationRecorder 会自动记录每轮对话:
- `user_message`: 用户发送的文本
- `retrieved_contexts`: 注入的状态上下文 (Markdown 表格)
- `ai_response`: LLM 返回的完整回复
- `state_before` / `state_after`: 对话前后的 `default.json` 快照

数据保存在 `data/evaluation/sessions/{session_id}.jsonl`

### Step 2: 运行评估脚本
```bash
python scripts/ragas_evaluation.py --session latest
```

### Step 3: 核心指标
| 指标 | 含义 | 目标分数 |
|:---|:---|:---|
| **Faithfulness** | AI 回答是否基于注入的状态上下文 | >= 0.80 |
| **Answer Relevancy** | AI 回答是否切题 | >= 0.85 |
| **Context Recall** | 状态注入是否覆盖了必要信息 | >= 0.75 |
| **Context Precision** | 注入的上下文信噪比 | >= 0.70 |

### Step 4: 查看报告
- CSV: `reports/ragas_report.csv`
- Markdown: `reports/ragas_report.md`

## RP 测试场景模板

### 背包管理场景
```
1. "我在新手村买了一把木剑"         -> expected: inventory += 木剑
2. "村长送了我3瓶红色药水"           -> expected: inventory += 红色药水x3
3. "我喝了一瓶红色药水"             -> expected: 红色药水 数量-1
4. "查看我的背包"                   -> expected: AI 提到木剑和药水
```

### 状态记忆场景
```
1. "我叫 Gabby，是一个新手冒险者"    -> expected: hero.name = Gabby
2. "我的名字是什么？"               -> expected: AI 回答 Gabby
```

## 自定义指标: State Consistency
超越标准 RAGAS，专门验证有状态 Agent 的记忆准确性:
- `inventory_accuracy`: 背包物品匹配率
- `state_update_recall`: 状态更新召回率
- `hallucination_rate`: 状态幻觉率 (越低越好)

运行: `python scripts/state_consistency_check.py --scenario backpack`
