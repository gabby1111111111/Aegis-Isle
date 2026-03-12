# 🌙 夜间自动化报告

> **运行时间**: 2026-03-12 05:55:16
> **运行环境**: Windows / Python 3.10.11

---

## 🧪 测试结果

| 检查项 | 状态 | 详情 |
|--------|------|------|
| Pytest | ❌ | 失败 |
| Flake8 | ⚠️ | 有警告 |
| 核心导入 | ✅ | 全部通过 |

### Pytest 详情
```
命令超时 (180s): C:\Users\MR\AppData\Local\Programs\Python\Python310\python.exe -m pytest tests/ -v --tb=short -q
```

### Flake8 详情
```
9     E722 do not use bare 'except'
64    F401 'json' imported but unused
12    F541 f-string is missing placeholders
1     F811 redefinition of unused 'admin_router' from line 13
1     F821 undefined name 'torch'
5     F841 local variable 'sender_id' is assigned to but never used
3     W291 trailing whitespace
34    W292 no newline at end of file
327   W293 blank line contains whitespace
603
```

---

## 📊 变更统计

- 最近 24h commits: **2**
- Aegis 架构变更: **2** 文件
- Agent 管理变更: **2** 文件
- 面试材料变更: **0** 文件

---

## 🐙 Github 子模块连线盘点

| 子项目 | 本地坐标 | 同步与仓库状态 |
|--------|----------|------|
| [Aegis-Isle 主项目](https://github.com/gabby1111111111/Aegis-Isle) | `E:/Aegis_Isle/AegisIsle_cc_ver/Aegis-Isle` | ✅ 已关联 Git，有 18 个未提交变更 |
| [Love & Code 面试](https://github.com/gabby1111111111/Love-and-Code-Interview) | `E:/Love-and-Code-Interview` | ✅ 已关联 Git，有 2 个未提交变更 |
| [ST-Companion-Link](https://github.com/gabby1111111111/ST-Companion-Link-Suite) | `E:/ST-Companion-Link` | ✅ 已关联 Git，有 0 个未提交变更 |
| [世界线管理器](https://github.com/gabby1111111111/Universe-Manager) | `E:/universe_manager` | ✅ 已关联 Git，有 2 个未提交变更 |
| [Bubby 品牌总管](https://github.com/gabby1111111111/bubby-and-premitted-land) | `C:/Users/MR/Desktop/bubby report` | ✅ 已关联 Git，有 1 个未提交变更 |

---

## 📝 更新的文件
- `cowokers_ai/CURRENT_TASK.md` — 三轨看板已更新
- `cowokers_ai/interview_changelog.md` — 面试材料已同步
- `logs/nightly/{datetime.now().strftime('%Y%m%d')}.log` — 详细日志

---

*此报告由 nightly_pipeline.py 自动生成*
