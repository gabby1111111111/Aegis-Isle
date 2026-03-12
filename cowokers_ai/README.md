# 🏢 cowokers_ai — AI 协作中心

> Gabby 大人管理 Agent 团队的指挥部。你只需要看这 **6 个核心文件**。

## 📌 核心文件（你需要关注的）

| 文件 | 什么时候看 | 内容 |
|------|-----------|------|
| **CURRENT_TASK.md** | ⭐ 每天第一个看 | 三轨看板：当前在做什么、接下来做什么 |
| **ROADMAP.md** | 每周看一次 | 项目全局进度和里程碑 |
| **REVIEW_PACKAGE.md** | Agent 推送通知后看 | 最新一个等你验收的功能证据包 |
| **AGENTS.md** | 新 Agent 上岗时看 | Agent 甲/乙/丙的身份和分工 |
| **NIGHT_SHIFT_RULES.md** | 不需要主动看 | 夜班 Agent 的行为准则 |
| **interview_changelog.md** | 面试前翻翻 | 代码变更 → 面试话术的映射 |

## 📁 子文件夹

| 文件夹 | 内容 | 你需要看吗？ |
|--------|------|-------------|
| `reports/` | QA 测试报告、夜间报告、flake8 基线 | 出 Bug 时翻 |
| `plans/` | 设计文档、架构可行性分析、集成方案 | 审批设计时看 |
| `archive/` | 已完成的 Agent 交付报告、Demo 脚本 | 基本不用看 |
| `auto_generated_docs/` | 自动生成的 API 文档 | 基本不用看 |

## 🔔 日常工作流

```
1. 手机收到 ntfy 推送 → 打开 REVIEW_PACKAGE.md → 按"验收指南"操作
2. 每天早上看 CURRENT_TASK.md 了解进度
3. 需要安排新任务 → 编辑 CURRENT_TASK.md 或直接跟 Agent 说
```
