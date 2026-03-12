---
description: 夜间自动化管线 - 测试、看板更新、面试同步
---

# 夜间自动管线 (Nightly Pipeline)

> 你睡觉时自动运行：跑测试 → 更新看板 → 同步面试材料

// turbo-all

## 0. 启动前置侦察（过去 6 小时回溯）

在流水线启动前，先查一次过去 6 小时内发生了什么，确认目前是在为哪些新鲜的改动而执行夜间测试：
```powershell
git status -s
git log --all --since="6 hours ago" --name-status
```

## 完整运行（推荐每晚定时执行）

1. 运行完整夜间管线
```
python scripts/nightly_pipeline.py
```

2. 检查生成的报告
```
cat cowokers_ai/reports/NIGHTLY_REPORT.md
```

3. 查看三轨看板是否更新
```
cat cowokers_ai/CURRENT_TASK.md
```

4. 查看面试材料同步结果
```
cat cowokers_ai/interview_changelog.md
```

---

## 单独模式

### 只跑测试
```
python scripts/nightly_pipeline.py --test-only
```

### 只同步面试材料
```
python scripts/nightly_pipeline.py --sync-resume
```

---

## 设置 Windows 定时任务

在 PowerShell 中执行以下命令，设置每晚 2:00 自动运行：

```powershell
$action = New-ScheduledTaskAction -Execute "python" -Argument "E:\Aegis_Isle\AegisIsle_cc_ver\Aegis-Isle\scripts\nightly_pipeline.py" -WorkingDirectory "E:\Aegis_Isle\AegisIsle_cc_ver\Aegis-Isle"
$trigger = New-ScheduledTaskTrigger -Daily -At 2:00AM
Register-ScheduledTask -TaskName "AegisNightly" -Action $action -Trigger $trigger -Description "Aegis-Isle 夜间自动管线"
```

---

## 5. 强制门禁：视觉验证 (Pre-Review Gate)

> [!CAUTION]
> 如果夜间管线中有 Agent 做了**功能性代码改动**（新增端点、修改 UI、修复 Bug），
> 必须在提交前激活 `.agent/skills/pre-review-gate/SKILL.md`，走完 Gate 4 + Gate 5。

```
# 1. 如果改动涉及 Streamlit 页面:
streamlit run <改动的页面>.py --server.port <空闲端口>
# → 用 browser_subagent 截图 + 录屏核心交互

# 2. 如果改动涉及 FastAPI 端点:
# → 用 browser_subagent 访问 /docs，截图 Swagger UI + 调用结果

# 3. 生成证据包:
# → 写 cowokers_ai/REVIEW_PACKAGE.md，嵌入截图/录屏路径
```

纯 lint/格式化修复（如 `auto-review-loop` 产出）可以豁免 Gate 4，但仍需通过 Gate 1-3。

---

## 管线产出文件

| 输出 | 路径 | 说明 |
|------|------|------|
| 夜间报告 | `cowokers_ai/reports/NIGHTLY_REPORT.md` | 测试结果 + 变更统计 |
| 三轨看板 | `cowokers_ai/CURRENT_TASK.md` | 按三条战线自动分类 |
| 面试同步 | `cowokers_ai/interview_changelog.md` | 代码变更 → 面试话术 |
| 详细日志 | `logs/nightly/YYYYMMDD_HHMMSS.log` | 历史日志存档 |
| **📦 证据包** | **`cowokers_ai/REVIEW_PACKAGE.md`** | **截图 + 录屏 + 验证报告** |
