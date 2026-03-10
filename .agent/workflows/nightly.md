---
description: 夜间自动化管线 - 测试、看板更新、面试同步
---

# 夜间自动管线 (Nightly Pipeline)

> 你睡觉时自动运行：跑测试 → 更新看板 → 同步面试材料

// turbo-all

## 完整运行（推荐每晚定时执行）

1. 运行完整夜间管线
```
python scripts/nightly_pipeline.py
```

2. 检查生成的报告
```
cat cowokers_ai/NIGHTLY_REPORT.md
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

## 管线产出文件

| 输出 | 路径 | 说明 |
|------|------|------|
| 夜间报告 | `cowokers_ai/NIGHTLY_REPORT.md` | 测试结果 + 变更统计 |
| 三轨看板 | `cowokers_ai/CURRENT_TASK.md` | 按三条战线自动分类 |
| 面试同步 | `cowokers_ai/interview_changelog.md` | 代码变更 → 面试话术 |
| 详细日志 | `logs/nightly/YYYYMMDD_HHMMSS.log` | 历史日志存档 |
