---
name: dual-agent-qa
description: 双 Agent 开发-质检循环 — 一个写代码一个 Review+测试，循环到质检官满意为止
---

# 双 Agent 质检循环 (Dual Agent QA Loop)

## 概述

本 Skill 定义了一种**双 Agent 协作流程**：
- **Agent DEV（开发者）**：负责写代码、修 Bug、实现功能
- **Agent QA（质检官）**：负责跑测试、静态分析、代码 Review，给出 PASS 或 FAIL 判定

循环直到 QA Agent 给出 **PASS** 为止。所有改动必须在独立分支上进行。

---

## 工作流程

```
┌──────────────┐
│  Step 0      │  QA Agent 读 project_mapping.yaml + ROADMAP.md
│  入场检查    │  确认当前模块有哪些文件
└──────┬───────┘
       ▼
┌──────────────┐
│  Step 1      │  DEV Agent 在新分支上做开发/修改
│  开发        │  commit 后通知 QA
└──────┬───────┘
       ▼
┌──────────────────────────────────────────────┐
│  Step 2: QA 质检循环（最多 5 轮）             │
│                                               │
│  2a. 运行 pytest tests/ -v                    │
│      → 全绿才继续，否则立即 FAIL              │
│                                               │
│  2b. 运行 flake8 src/ --count                 │
│      → 新增告警数 ≤ 0 才继续                  │
│                                               │
│  2c. 代码 Review（QA 逐文件读改动）            │
│      检查清单：                                │
│      □ 函数是否有 docstring                    │
│      □ 异步函数是否正确使用 await              │
│      □ 新增端点是否有错误处理 try/except       │
│      □ 是否有硬编码路径/密钥                   │
│      □ 新增依赖是否在 requirements.txt 中      │
│      □ 命名是否一致（snake_case）              │
│      □ 是否影响了现有 API 的兼容性             │
│                                               │
│  2d. QA 判定：                                 │
│      PASS → 结束循环，进入 Step 3              │
│      FAIL → 写 REVIEW_FEEDBACK.md              │
│              列出所有问题                      │
│              → 回到 Step 1 让 DEV 修           │
│                                               │
│  ⚠️ 最多循环 5 次。第 5 次仍然 FAIL 则：       │
│     写 ESCALATION.md 标记需要人工介入          │
└──────────────────────────────────────────────┘
       ▼
┌──────────────┐
│  Step 3      │  QA 写最终 QA_REPORT.md
│  出场报告    │  包含：测试通过数、告警变化、
│              │  Review 结论、循环次数
└──────────────┘
```

---

## Step 0: 入场检查

QA Agent 必须先理解项目结构：

```bash
# 1. 读项目地图
cat .agent/project_mapping.yaml

# 2. 读 ROADMAP 了解当前进度
cat cowokers_ai/ROADMAP.md

# 3. 记录当前测试基线
pytest tests/ -v 2>&1 | tail -5    # 记录当前 passed 数
flake8 src/ --count 2>&1 | tail -1  # 记录当前告警数
```

将基线数据写入 `cowokers_ai/reports/QA_BASELINE.md`：
```markdown
# QA 基线 - {日期}
- pytest: XX passed, XX warnings
- flake8: XX 条告警
- 当前分支: main @ {commit_hash}
```

---

## Step 1: DEV Agent 开发

DEV Agent 的规则：
1. 从 main 创建新分支：`git checkout -b feature/xxx` 或 `auto-fix/xxx`
2. 完成开发后自己先跑一遍 `pytest` 确保不爆红
3. `git add . && git commit -m "描述"`
4. 在 `cowokers_ai/DEV_READY.md` 写一句："开发完成，请 QA 检查分支 `xxx`"

---

## Step 2: QA 质检循环

### 2a. 自动化测试

```bash
# 跑 pytest
pytest tests/ -v --tb=short

# 如果有新增模块，检查是否有对应测试文件
# 例如新增了 src/aegis_isle/api/routers/diary_review.py
# 就应该有 tests/test_diary_review.py
```

**判定标准**：
- ✅ 所有测试 PASSED（允许和基线相同数量的 warnings）
- ❌ 任何 FAILED → 立即判定 FAIL

### 2b. 静态分析

```bash
# 跑 flake8
flake8 src/ --count --statistics

# 对比基线
# 新增告警数 = 当前告警数 - 基线告警数
```

**判定标准**：
- ✅ 新增告警数 ≤ 0（不允许引入新告警）
- ⚠️ 新增 1-3 条 → 警告但不立即 FAIL，计入 Review
- ❌ 新增 > 3 条 → FAIL

### 2c. 代码 Review

QA Agent 必须 **逐文件阅读** DEV Agent 的改动（`git diff main..当前分支`）。

**检查清单**（每项勾选）：

```markdown
## Review Checklist - {文件名}
- [ ] 函数签名和返回类型是否明确
- [ ] 是否有 docstring 说明功能
- [ ] 异步函数是否正确使用 await（不能漏掉）
- [ ] 新增端点是否有 try/except 错误处理
- [ ] 是否有硬编码的绝对路径或密钥
- [ ] 新增 import 是否在 requirements.txt 中
- [ ] 命名风格是否与项目一致（snake_case）
- [ ] 是否影响了现有 API 的入参/出参兼容性
- [ ] 是否有潜在的并发/线程安全问题
- [ ] 日志级别是否合理（不要用 print，用 logger）
```

### 2d. 判定与反馈

**如果 PASS**：
```markdown
# QA 判定: ✅ PASS
轮次: 第 X 轮
分支: feature/xxx
所有测试通过，无新增告警，代码 Review 无重大问题。
```

**如果 FAIL**：
在 `cowokers_ai/REVIEW_FEEDBACK.md` 写具体问题：
```markdown
# QA 反馈 - 第 X 轮
## ❌ 需要修复
1. `src/xxx.py:42` — 缺少错误处理，如果 FAISS 索引不存在会崩溃
2. `src/xxx.py:78` — `await` 漏掉了，会导致协程未执行
## ⚠️ 建议改进（不阻塞）
1. `src/xxx.py:15` — 建议把 magic number 5000 提取为常量
```

然后 DEV Agent 读这份反馈，修改后再提交，QA 重新检查。

---

## Step 3: 出场报告

最终 QA 通过后，写 `cowokers_ai/reports/QA_REPORT_{模块名}.md`：

```markdown
# QA 报告 - {模块名}
- 日期: YYYY-MM-DD
- 分支: feature/xxx
- 循环轮次: X 轮
- 最终 pytest: XX passed, XX warnings
- 最终 flake8: XX 条（基线 XX 条，新增 0 条）
- Review 结论: PASS
- 备注: {任何需要 Gabby 大人知道的事}
```

---

## 使用方式

### 单人模式（一个 Agent 同时扮演 DEV 和 QA）

如果只有一个 Agent 窗口，它可以自己交替扮演两个角色：
1. 先以 DEV 身份写代码
2. 再切换为 QA 身份跑测试 + Review
3. 发现问题后切回 DEV 修复

### 双人模式（两个 Agent 窗口）

**给 DEV Agent 的指令**：
```
请读取 E:\Aegis_Isle\AegisIsle_cc_ver\Aegis-Isle\.agent\skills\dual-agent-qa\SKILL.md，
按照 Step 1 的规则，在新分支上完成以下任务：
{具体任务描述}
做完后在 cowokers_ai/DEV_READY.md 写一句通知 QA。
```

**给 QA Agent 的指令**：
```
请读取 E:\Aegis_Isle\AegisIsle_cc_ver\Aegis-Isle\.agent\skills\dual-agent-qa\SKILL.md，
按照 Step 0 建立基线，然后按照 Step 2 对分支 {分支名} 进行质检循环。
如果 FAIL 就写 REVIEW_FEEDBACK.md 等 DEV 修，最多循环 5 次。
最终写 QA_REPORT.md。
```

---

## 按模块逐个质检的推荐顺序

根据 project_mapping.yaml，建议按以下顺序逐模块质检：

| 顺序 | 模块 | 涉及文件 | 优先级 |
|:----:|------|---------|:------:|
| 1 | Shujuku 状态管理 | `core/state/*.py` (8 个文件) | 🔴 高 |
| 2 | RAG 记忆引擎 | `rag/*.py` (8 个文件) | 🔴 高 |
| 3 | API 网关路由 | `api/routers/*.py` (5 个文件) | 🔴 高 |
| 4 | CharLifeAgent | `agents/*.py` (5 个文件) | 🟡 中 |
| 5 | 面试系统 | `interview/*.py` (7 个文件) | 🟡 中 |
| 6 | 前端文件 | `frontend/*.py` + `st_extension/` | 🟢 低 |
| 7 | 脚本 | `scripts/*.py` | 🟢 低 |

每个模块质检通过后，才进入下一个模块。
