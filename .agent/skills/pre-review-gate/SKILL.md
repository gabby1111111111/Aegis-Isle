---
name: pre-review-gate
description: 强制验收门禁 — Agent 提交 Review 前必须自证代码能跑、页面能看、截图为证
---

# 🚪 Pre-Review Quality Gate (强制验收门禁)

> **铁律**: 在使用 `notify_user` 请求 Gabby 大人 Review 之前，**必须**走完以下 5 道关卡。
> 任何一道未通过，**禁止**提交。半成品 = 浪费 Boss 时间 = 扣分。

---

## 适用范围

当你完成了以下任何一种工作，且准备提交给用户 Review 时，**必须激活本 Skill**：

- 新增/修改了 Streamlit 页面
- 新增/修改了 FastAPI 端点
- 新增/修改了纯后端逻辑（模型、管线、Agent）
- 修复了一个 Bug 并声称已修好

**不适用**：纯文档编写、纯设计讨论、`.md` 文件更新。

---

## 🏗️ 五道关卡

```
Gate 1  pytest 全绿          ──┐
Gate 2  flake8 零新增告警     ──┤ 代码层 (复用 dual-agent-qa)
Gate 3  代码 Review Checklist ──┘
Gate 4  浏览器实测 + 多张截图  ──── 视觉层 (本 Skill 新增)
Gate 5  生成 REVIEW_PACKAGE   ──── 证据层 (本 Skill 新增)
```

---

### Gate 1: 自动化测试 ✅

```bash
python -m pytest tests/ -v --tb=short
```

**通过标准**: 0 FAILED。允许 warnings 但不允许新增 FAILED。
**失败处理**: 修到全绿再继续，最多 3 轮 (参考 `auto-review-loop` 的安全阀 2)。

---

### Gate 2: 静态分析 ✅

```bash
flake8 src/ --count --statistics
```

**通过标准**: 新增告警数 ≤ 0（对比基线）。
**失败处理**: 修到不新增告警。涉及核心模块逻辑的打 `[🚨 需 Gabby 亲自定夺]` 标签跳过。

---

### Gate 3: 代码 Review Checklist ✅

对你自己改动的每个文件，逐项勾选（参考 `dual-agent-qa` Step 2c）：

```markdown
- [ ] 函数签名和返回类型明确
- [ ] 有 docstring
- [ ] 异步函数正确使用 await
- [ ] 新端点有 try/except
- [ ] 无硬编码路径/密钥
- [ ] 新 import 在 requirements.txt 中
- [ ] 命名一致 (snake_case)
- [ ] 不破坏现有 API 兼容性
```

**通过标准**: 无阻塞项。
**失败处理**: 自己修，不要把明显问题留给 Boss。

---

### Gate 4: 浏览器实测 + 多张截图 📸 (核心新增)

**这是本 Skill 的灵魂关卡。** 根据改动类型选择对应的验证方式：

> [!CAUTION] 
> 🚨 **防死锁与防崩溃指引 (Anti-Deadlock Fallback)**
> 由于开启浏览器并截图极其消耗系统资源，多特工同时进行或遇到白屏页面时极易导致本地内存挤爆卡死。
> - **最多重试 3 次**：如果你调用的 `browser_subagent` 连续 3 次拿不到有用截图或报错。
> - **立即放弃截图**：绝对不要在一个错误页面上死磕死循环！
> - **降级处理**：如果你确信自己的代码没问题，只是截图组件挂了，允许你直接跳过截图。在 `REVIEW_PACKAGE.md` 里写上 `[⚠️ Gate 4 视觉验证失败：页面截取陷入死锁，已降级]`，并用纯文字描述你的验证步骤。然后进入下一关交卷。

#### 4A. 涉及 Streamlit 前端

1. 启动服务:
   ```bash
   streamlit run <改动的页面>.py --server.port <空闲端口>
   ```
2. 使用 `browser_subagent` 工具:
   - 打开页面 URL
   - **截图**操作核心交互流程（点按钮、输入数据、切换 Tab 等）的各个关键步骤
   - 必须提供至少 2-3 张截图以证明流程连贯性
3. 验证:
   - 页面无报错红框
   - 核心功能按钮可点击且有响应
   - 数据正确显示

#### 4B. 涉及 FastAPI 后端

1. 启动服务（如未运行）:
   ```bash
   uvicorn src.aegis_isle.api.main:app --port 8001
   ```
2. 使用 `browser_subagent` 工具:
   - 打开 `http://localhost:8001/docs` (Swagger UI)
   - 截图确认新端点存在
   - 在 Swagger UI 中实际调用新端点
   - 截图请求和响应结果
3. 验证:
   - 端点返回预期状态码
   - 响应 JSON 结构正确

#### 4C. 纯后端逻辑（无 UI）

1. 在终端运行相关 pytest 并截取输出:
   ```bash
   python -m pytest tests/test_<相关模块>.py -v
   ```
2. 截图终端输出的测试结果
3. 如有日志输出，截图关键日志行

**通过标准**: 截图中无报错，核心功能可正常运行，步骤连贯。
**失败处理**: 修到能跑为止，然后重新截图。

---

### Gate 5: 生成 REVIEW_PACKAGE.md 📦

在 `cowokers_ai/` 目录下生成证据包文件：

```markdown
# 📦 Review 证据包 — {功能名称}

- 日期: YYYY-MM-DD
- 分支: feature/xxx
- Agent: {Agent 名称}

## ✅ Gate 1: 自动化测试
- pytest: XX passed, 0 failed, XX warnings
- 命令: `python -m pytest tests/ -v --tb=short`

## ✅ Gate 2: 静态分析
- flake8 新增告警: 0 条
- 基线: XXX 条 → 当前: XXX 条

## ✅ Gate 3: 代码 Review
- [x] docstring ✓
- [x] await 正确 ✓
- [x] 错误处理 ✓
- [x] 无硬编码 ✓
- [x] 命名一致 ✓
- [x] API 兼容 ✓

## ✅ Gate 4: 视觉验证
- **启动命令**: 
  > 证明服务是怎么起起来的，例：`streamlit run app.py` / `uvicorn main:app`
- **测试步骤**: 
  > 描述你具体点了什么，例：1. 点击左侧边栏; 2. 输入“测试”; 3. 点击提交
- **达标标准**: 
  > 你用什么指标判断功能是正常的？例：页面成功显示了返回结果，且控制台 0 报错。

**验证截图证明**:
![起始页面截图](起始截图绝对路径)
![操作后页面截图](操作后截图绝对路径)
> 附加描述: "如图所示，XXX 功能已按预期显示..."

## 🏁 结论
所有 5 道关卡全部通过 ✅
请 Gabby 大人验收。
```

然后在 `notify_user` 中：
- `PathsToReview` 设为 `REVIEW_PACKAGE.md` 的路径
- `Message` 简要说明改了什么，**不要复述证据包内容**

### 📱 最终步：呼叫 Gabby 大人 (Mobile Push)

在调用 `notify_user` 的同时（或之前），你必须**主动往 Gabby 大人的专属手机频道发一条 PUSH 通知**，提醒她来看你的成果。

执行以下命令：
```bash
curl.exe -d "Gabby大人！我的证据包已经准备好了，快来看！✨" -H "Title: 📦 Review 邀请 - {功能名称}" -H "Priority: high" -H "Tags: bell,package" "https://ntfy.sh/gabby-ring"
```

---

## ⚡ 速查流程图

```
你写完代码了
    │
    ▼
 pytest 全绿？ ──否──→ 修代码 → 重跑
    │是
    ▼
 flake8 零新增？ ──否──→ 修代码 → 重跑
    │是
    ▼
 Review Checklist 全勾？ ──否──→ 修代码
    │是
    ▼
 启动服务 + 浏览器多张截图 ──报错──→ 修代码 → 重新截图
    │正常
    ▼
 生成 REVIEW_PACKAGE.md
    │
    ▼
 notify_user（附带证据包）
    │
    ▼
 Gabby 大人看到的是：成品 + 证据 ✅
```

---

## 🚨 违规处罚

如果 Agent 跳过本 Skill 直接提交 Review，且 Gabby 大人发现是半成品：
1. 该功能退回重做
2. 必须在 `REVIEW_PACKAGE.md` 中额外写一段"为什么上次漏检了"的反思
3. 下次提交时 Gate 4 必须包含**至少 3 张连贯步骤的截图**

---

## 召唤咒语

当你准备提交 Review 时，在脑海中默念：

```
我是否走完了 5 道门禁？
我有截图证明功能能跑吗？
Gabby 大人看到这个会说"又是半成品"吗？
```

如果第三个问题的答案是"会"，那就**不要提交**。
