"""
Aegis-Isle 夜间自动管线 (Nightly Pipeline)
==========================================
设计目标：你睡觉的时候，它自动做三件事：
  1. 跑测试 + review 代码质量
  2. 更新三轨任务看板 (CURRENT_TASK.md)
  3. 根据代码变更同步面试材料

用法：
  python scripts/nightly_pipeline.py              # 完整管线
  python scripts/nightly_pipeline.py --test-only   # 只跑测试
  python scripts/nightly_pipeline.py --sync-resume # 只同步面试
  
Windows 计划任务设置 (每晚 2:00 自动运行):
  schtasks /create /tn "AegisNightly" /tr "python E:\\Aegis_Isle\\AegisIsle_cc_ver\\Aegis-Isle\\scripts\\nightly_pipeline.py" /sc daily /st 02:00

作者: Nightly Bot 🤖
"""

import subprocess
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# === 路径配置 ===
PROJECT_ROOT = Path(__file__).parent.parent
COWORKERS_DIR = PROJECT_ROOT / "cowokers_ai"
LOGS_DIR = PROJECT_ROOT / "logs" / "nightly"
TESTS_DIR = PROJECT_ROOT / "tests"
SRC_DIR = PROJECT_ROOT / "src"
REPORT_PATH = COWORKERS_DIR / "NIGHTLY_REPORT.md"
DASHBOARD_PATH = COWORKERS_DIR / "CURRENT_TASK.md"
INTERVIEW_NOTES_PATH = COWORKERS_DIR / "interview_changelog.md"


def ensure_dirs():
    """确保必要的目录存在"""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    COWORKERS_DIR.mkdir(parents=True, exist_ok=True)


def run_cmd(cmd: list[str], cwd: Optional[Path] = None, timeout: int = 300) -> tuple[int, str, str]:
    """运行命令并返回 (returncode, stdout, stderr)"""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd or PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"命令超时 ({timeout}s): {' '.join(cmd)}"
    except Exception as e:
        return -1, "", f"命令执行失败: {e}"


# ============================================================
# 阶段 1: 自动测试 + 代码质量检查
# ============================================================

def phase_test_and_review() -> dict:
    """跑测试、lint、生成质量报告"""
    print("\n" + "=" * 60)
    print("🧪 阶段 1: 自动测试 & 代码审查")
    print("=" * 60)

    report = {
        "pytest": {"passed": False, "detail": ""},
        "flake8": {"passed": False, "detail": ""},
        "import_check": {"passed": False, "detail": ""},
    }

    # 1.1 Pytest
    print("\n  📋 运行 pytest...")
    code, out, err = run_cmd(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short", "-q"],
        timeout=180,
    )
    report["pytest"]["passed"] = (code == 0)
    report["pytest"]["detail"] = out[-2000:] if out else err[-2000:]
    print(f"  {'✅' if code == 0 else '❌'} pytest: returncode={code}")

    # 1.2 Flake8 (只检查 src/)
    print("  📋 运行 flake8...")
    code, out, err = run_cmd(
        [sys.executable, "-m", "flake8", "src/", "--max-line-length=120", "--count", "--statistics"],
        timeout=60,
    )
    report["flake8"]["passed"] = (code == 0)
    # 只取最后几行统计数据
    lines = (out or err).strip().split("\n")
    report["flake8"]["detail"] = "\n".join(lines[-10:])
    print(f"  {'✅' if code == 0 else '⚠️'} flake8: {lines[-1] if lines else 'N/A'}")

    # 1.3 关键模块可导入性检查
    print("  📋 检查核心模块可导入性...")
    import_code = """
import sys; sys.path.insert(0, 'src')
try:
    from aegis_isle.rag.embedder import get_embedder
    from aegis_isle.rag.st_memory_manager import STMemoryManager
    from aegis_isle.core.state.manager import StateManager
    print("ALL_IMPORTS_OK")
except Exception as e:
    print(f"IMPORT_FAIL: {e}")
"""
    code, out, err = run_cmd(
        [sys.executable, "-c", import_code],
        timeout=60,
    )
    ok = "ALL_IMPORTS_OK" in out
    report["import_check"]["passed"] = ok
    report["import_check"]["detail"] = out.strip() if out else err.strip()
    print(f"  {'✅' if ok else '❌'} 核心模块导入: {'全部通过' if ok else out or err}")

    return report


# ============================================================
# 阶段 2: Git 变更分析 + 三轨看板更新
# ============================================================

def phase_update_dashboard() -> dict:
    """分析 Git 日志，更新三轨进度看板"""
    print("\n" + "=" * 60)
    print("📊 阶段 2: 三轨看板更新")
    print("=" * 60)

    changes = {
        "recent_commits": [],
        "files_changed": [],
        "track_aegis": [],
        "track_agent": [],
        "track_interview": [],
    }

    # 2.1 获取最近 24h 的 commit
    since_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    code, out, err = run_cmd(
        ["git", "log", f"--since={since_date}", "--oneline", "--no-merges", "-20"],
    )
    if code == 0 and out.strip():
        changes["recent_commits"] = out.strip().split("\n")
    print(f"  📝 最近 24h commits: {len(changes['recent_commits'])} 条")

    # 2.2 获取变更文件列表
    code, out, err = run_cmd(
        ["git", "diff", "--name-only", "HEAD~5", "HEAD"],
    )
    if code == 0 and out.strip():
        changes["files_changed"] = out.strip().split("\n")

    # 2.3 按路径分类到三轨
    for f in changes["files_changed"]:
        f_lower = f.lower()
        if any(kw in f_lower for kw in ["rag/", "agents/", "state/", "api/", "memory", "faiss", "embedder", "daily_digest", "event_logger"]):
            changes["track_aegis"].append(f)
        elif any(kw in f_lower for kw in [".agent/", "workflow", "skill", "cowokers_ai/agents"]):
            changes["track_agent"].append(f)
        elif any(kw in f_lower for kw in ["interview", "resume", "demo", "love_and_code"]):
            changes["track_interview"].append(f)

    print(f"  🏗️  Aegis 架构变更: {len(changes['track_aegis'])} 文件")
    print(f"  🤖 Agent 管理变更: {len(changes['track_agent'])} 文件")
    print(f"  📝 面试准备变更: {len(changes['track_interview'])} 文件")

    # 2.4 统计项目健康指标
    code, out, err = run_cmd(
        [sys.executable, "-c", f"import glob; print(len(glob.glob(r'{PROJECT_ROOT}/tmp_*')))"],
    )
    tmp_count = int(out.strip()) if code == 0 and out.strip().isdigit() else "?"

    # 2.5 生成看板 Markdown
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    dashboard_content = f"""# 🎯 三轨任务追踪看板

> **自动更新时间**: {now} | **更新方式**: nightly_pipeline.py 自动生成

---

## 🏗️ Track 1: Aegis 架构演进
- **最近变更文件数**: {len(changes['track_aegis'])}
- **核心模块**:
{chr(10).join(f'  - `{f}`' for f in changes['track_aegis'][:10]) or '  - (最近无变更)'}
- **里程碑**: V6 统一日记系统 → Demo 验证
- **下一步**: 按 Demo 脚本预演，确认功能流转

## 🤖 Track 2: Antigravity 多 Agent 管理
- **最近变更文件数**: {len(changes['track_agent'])}
- **核心模块**:
{chr(10).join(f'  - `{f}`' for f in changes['track_agent'][:10]) or '  - (最近无变更)'}
- **里程碑**: Skills + Workflows 体系搭建
- **下一步**: 完善 nightly 自动化管线，扩充 Skills

## 📝 Track 3: 面试简历准备
- **最近变更文件数**: {len(changes['track_interview'])}
- **核心模块**:
{chr(10).join(f'  - `{f}`' for f in changes['track_interview'][:10]) or '  - (最近无变更)'}
- **里程碑**: 简历润色 + Demo 录制
- **下一步**: 查看 `interview_changelog.md` 中的自动摘要

---

## 📈 项目健康度
| 指标 | 值 |
|------|-----|
| 最近 24h commits | {len(changes['recent_commits'])} |
| 根目录 tmp 文件数 | {tmp_count} |
| 总变更文件数 | {len(changes['files_changed'])} |

## 📜 最近 Commits
{chr(10).join(f'- `{c}`' for c in changes['recent_commits'][:10]) or '- (无)'}
"""

    DASHBOARD_PATH.write_text(dashboard_content, encoding="utf-8")
    print(f"  ✅ 看板已更新: {DASHBOARD_PATH}")

    return changes


# ============================================================
# 阶段 3: 代码变更 → 面试材料同步 (Gabriella 特别版)
# ============================================================

# 核心映射表: 代码路径关键词 → 面试题域 & 赛博茶话会翻译 & 面试官话术
INTERVIEW_MODULE_MAP = {
    "st_memory": {
        "label": "SillyTavern 记忆管理",
        "code_path": "src/aegis_isle/rag/st_memory_manager.py",
        "interview_zone": "RAG 多宇宙检索",
        "gabby_talk": "就像你在《博德之门3》里有 78 条世界线存档，每条都能独立读取回忆——这就是我们的多宇宙 FAISS 检索",
        "pro_talk": "我设计了基于 FAISS 的多宇宙向量检索系统，支持 78 个独立向量空间的跨宇宙联合召回，通过 asyncio 并发控制实现毫秒级响应",
    },
    "embedder": {
        "label": "向量嵌入引擎",
        "code_path": "src/aegis_isle/rag/embedder.py",
        "interview_zone": "Embedding 模型选型",
        "gabby_talk": "从 all-MiniLM (像翻译软件) 升级到 BGE-Large-zh (像母语者)，1024 维向量就是角色的'第六感精度'提升了好几个段位",
        "pro_talk": "完成了从 all-MiniLM-L6-v2 到 BAAI/bge-large-zh-v1.5 的全量迁移，维度从 384 提升到 1024，中文语义理解能力显著增强",
    },
    "daily_digest": {
        "label": "每日摘要聚合",
        "code_path": "src/aegis_isle/rag/daily_digest.py",
        "interview_zone": "自治记忆系统",
        "gabby_talk": "类似二次元角色每天睡前写日记——把白天浏览网页、聊天、刷题的事件自动压缩成一篇'今日心情总结'，存到 FAISS 里当长期记忆",
        "pro_talk": "实现了 DailyDigest 聚合引擎，每日自动从 JSONL 事件流中读取多源数据，配合 ECoT Prompt 调用 LLM 生成结构化日记摘要，写入独立的 diary FAISS 索引",
    },
    "event_logger": {
        "label": "事件溯源总线",
        "code_path": "src/aegis_isle/rag/event_logger.py",
        "interview_zone": "事件驱动架构",
        "gabby_talk": "LifeEventBus 就是角色的'生命监控摄像头'——ST聊天、面试做题、CharLifeAgent自己逛维基百科，三路数据像三条弹幕一样实时写入同一个 JSONL 流",
        "pro_talk": "构建了 LifeEventBus 事件总线，采用事件溯源模式统一接入 SillyTavern、Love&Code 面试系统、CharLifeAgent 三路数据源，通过 JSONL 追加写入保证数据完整性",
    },
    "char_life": {
        "label": "自治角色 Agent",
        "code_path": "src/aegis_isle/agents/",
        "interview_zone": "自治 Agent 架构",
        "gabby_talk": "CharLifeAgent 就像 NPC 在你不玩游戏的时候还会自己去探索地图——它主动搜维基百科、按角色兴趣整理信息，写进自己的日记",
        "pro_talk": "设计了 CharLifeAgent 自治 Agent，基于角色 Persona 主动执行信息检索（AgentFetch），通过严格 Prompt 控制 LLM 生成展示型自省日志，实现角色的离线认知增长",
    },
    "state/": {
        "label": "状态管理系统",
        "code_path": "src/aegis_isle/core/state/",
        "interview_zone": "结构化状态管理",
        "gabby_talk": "状态管理就是 RPG 的存档系统——每次操作自动快照，随时回滚。用 Pydantic 做数据校验就像装备栏不允许你把剑塞进药水格子里",
        "pro_talk": "设计了完整的 Pydantic 驱动状态管理系统，支持 Sheet 模型的 CRUD、自动快照与版本回滚。状态通过 XML 指令提取后异步持久化，不阻塞主请求链路",
    },
    "openai_compat": {
        "label": "OpenAI 兼容网关",
        "code_path": "src/aegis_isle/api/routers/openai_compat.py",
        "interview_zone": "API 网关设计",
        "gabby_talk": "我们伪装成 OpenAI 的 API——SillyTavern 以为自己在和 GPT 对话，其实背后是我们的整套 RAG 记忆注入 + 状态管理在暗中运作",
        "pro_talk": "实现了 OpenAI 兼容的 Chat Completions 网关，支持 SSE 流式输出。请求链路中同步注入 RAG 上下文和状态信息，LLM 响应后通过 BackgroundTask 异步提取状态变更",
    },
    "api/routers": {
        "label": "FastAPI 路由层",
        "code_path": "src/aegis_isle/api/routers/",
        "interview_zone": "后端架构",
        "gabby_talk": "路由层就是城堡的大门——memory 路由负责记忆搜索，openai_compat 路由负责伪装成 GPT 接口，所有请求都经过这里分流",
        "pro_talk": "FastAPI 异步路由层分离了对话生成（openai_compat）、记忆检索（memory）、状态管理 (state) 三大职责，通过 asyncio 并发控制保证高吞吐",
    },
    "graph_": {
        "label": "知识图谱检索",
        "code_path": "src/aegis_isle/rag/graph_searcher.py",
        "interview_zone": "多层级记忆架构",
        "gabby_talk": "graph_searcher 就是记忆的'关系网'——不只是搜关键词，而是像追星时扒偶像的社交关系图一样，找到记忆碎片之间的连接",
        "pro_talk": "知识图谱检索器实现了 sub_chunk → parent_chunk → episode 的多层级记忆召回，通过语义关联而非关键词匹配找到深层相关记忆",
    },
    "episode_": {
        "label": "剧情回忆检索",
        "code_path": "src/aegis_isle/rag/episode_searcher.py",
        "interview_zone": "Episode 上帝视角",
        "gabby_talk": "episode_searcher 就是看完一整季动漫后的'剧情总结'——不是一句一句回忆台词，而是提炼出每个故事弧的核心事件",
        "pro_talk": "Episode Searcher 从对话流中提取宏观叙事弧，生成上帝视角的剧情摘要，作为 RAG 长上下文的关键组成部分注入 LLM System Prompt",
    },
    "index.js": {
        "label": "ST 前端扩展",
        "code_path": "st_extension/aegis-memory/index.js",
        "interview_zone": "前端集成",
        "gabby_talk": "index.js 是我们潜入 SillyTavern 的'间谍程序'——通过 DOM Hook 和 CHAT_CHANGED 监听器，在不修改 ST 源码的情况下注入多宇宙选择 UI",
        "pro_talk": "实现了 SillyTavern 第三方扩展，通过 CHAT_CHANGED 事件监听自动识别角色，动态渲染多宇宙复选框 UI，采用非侵入式 DOM Hook 架构",
    },
    "interview": {
        "label": "面试系统集成",
        "code_path": "src/aegis_isle/interview/",
        "interview_zone": "Love & Code 面试系统",
        "gabby_talk": "面试系统通过 webhook 假装是 ST-Companion-Link 的数据——面试官问你 Python 题，AI角色以为你在'学习新技能'，这些刷题记录也变成了角色记忆的一部分",
        "pro_talk": "将 Love & Code 面试系统通过 webhook 伪装挂载到 ST-Companion-Link 的短期记忆缓冲区，实现面试数据与角色记忆的无缝融合",
    },
}


def phase_sync_interview(changes: dict):
    """根据代码变更，生成 Gabriella 特别版面试材料同步"""
    print("\n" + "=" * 60)
    print("📝 阶段 3: 面试材料同步 (Gabriella 特别版)")
    print("=" * 60)

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    aegis_files = changes.get("track_aegis", [])
    interview_files = changes.get("track_interview", [])
    all_changed = aegis_files + interview_files
    all_commits = changes.get("recent_commits", [])

    # 匹配变更文件到面试模块
    affected_modules = []
    for f in all_changed:
        f_lower = f.lower()
        for keyword, info in INTERVIEW_MODULE_MAP.items():
            if keyword in f_lower and info not in affected_modules:
                affected_modules.append(info)

    # 读取已有历史
    content_parts = []
    if INTERVIEW_NOTES_PATH.exists():
        existing = INTERVIEW_NOTES_PATH.read_text(encoding="utf-8")
        if "---\n\n##" in existing:
            history_start = existing.index("---\n\n##")
            content_parts.append(existing[history_start + 4:])

    # 构建新条目
    new_entry = f"""## [{now}] 自动扫描报告

### 🔄 代码变更摘要 (最近 24h)
- 共 **{len(all_commits)}** 个 commits
- Aegis 架构相关变更: **{len(aegis_files)}** 个文件
- 面试系统相关变更: **{len(interview_files)}** 个文件
"""

    if affected_modules:
        new_entry += "\n### 🎯 受影响的面试题域\n\n"
        new_entry += "| 模块 | 面试题域 | 需要复习 |\n"
        new_entry += "|------|----------|----------|\n"
        for mod in affected_modules:
            new_entry += f"| {mod['label']} | {mod['interview_zone']} | ⚠️ 代码已更新 |\n"

        new_entry += "\n### 🎮 赛博茶话会 · Gabby 的大白话翻译\n\n"
        for mod in affected_modules:
            new_entry += f"**{mod['label']}** (`{mod['code_path']}`)\n"
            new_entry += f"> 🐰 {mod['gabby_talk']}\n\n"

        new_entry += "### 👔 面试官视角 · 专业话术建议\n\n"
        for mod in affected_modules:
            new_entry += f"**Q: 请介绍你项目中{mod['interview_zone']}的设计**\n"
            new_entry += f"> 💼 {mod['pro_talk']}\n\n"
    else:
        new_entry += "\n### ℹ️ 本轮无面试相关模块变更，可沿用之前的面试亮点\n"

    new_entry += "\n"

    # 组装最终文件
    final_content = f"""# 📋 面试材料自动更新日志 (Gabriella 特别版)

> 此文件由 `nightly_pipeline.py` 自动维护
> 检测代码变更 → 定位面试题域 → 生成赛博茶话会翻译 + 专业面试话术
> **最新更新**: {now}

---

{new_entry}{''.join(content_parts)}"""

    INTERVIEW_NOTES_PATH.write_text(final_content, encoding="utf-8")
    print(f"  ✅ 面试材料已同步: {INTERVIEW_NOTES_PATH}")
    print(f"  📌 受影响的面试题域: {len(affected_modules)} 个")
    for mod in affected_modules:
        print(f"     → {mod['label']}: {mod['interview_zone']}")


# ============================================================
# 阶段 3.5: GitHub 子模块同步状态检查
# ============================================================
def phase_check_repos() -> list:
    """检查各个子模块的状态和与 Github 的连通性"""
    print("\n" + "=" * 60)
    print("🐙 阶段 3.5: Github 子模块同步检查")
    print("=" * 60)
    
    repos = [
        {"name": "Aegis-Isle 主项目", "path": "E:/Aegis_Isle/AegisIsle_cc_ver/Aegis-Isle", "url": "https://github.com/gabby1111111111/Aegis-Isle"},
        {"name": "Love & Code 面试", "path": "E:/Love-and-Code-Interview", "url": "https://github.com/gabby1111111111/Love-and-Code-Interview"},
        {"name": "ST-Companion-Link", "path": "E:/ST-Companion-Link", "url": "https://github.com/gabby1111111111/ST-Companion-Link-Suite"},
        {"name": "世界线管理器", "path": "E:/universe_manager", "url": "https://github.com/gabby1111111111/Universe-Manager"},
        {"name": "Bubby 品牌总管", "path": "C:/Users/MR/Desktop/bubby report", "url": "https://github.com/gabby1111111111/bubby-and-premitted-land"}
    ]
    
    results = []
    for repo in repos:
        p = Path(repo["path"])
        if not p.exists():
            status = "❌ 未找到本地路径"
        elif not (p / ".git").exists():
            status = f"⚠️ 未初始化 Git，(预留: [Repo]({repo['url']}))"
        else:
            # Check git status
            code, out, err = run_cmd(["git", "status", "-s"], cwd=p, timeout=10)
            if code == 0:
                uncommitted = len(out.strip().split("\n")) if out.strip() else 0
                status = f"✅ 已关联 Git，有 {uncommitted} 个未提交变更"
            else:
                status = "❌ Git 状态异常"
        
        print(f"  {repo['name']} -> {status}")
        results.append({"name": repo["name"], "path": repo["path"], "status": status, "url": repo["url"]})
        
    return results

# ============================================================
# 阶段 4: 汇总夜间报告
# ============================================================

def phase_generate_report(test_report: dict, changes: dict, repo_status: list = None):
    """生成完整的夜间运行报告"""
    print("\n" + "=" * 60)
    print("📄 阶段 4: 生成夜间报告")
    print("=" * 60)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    pytest_icon = "✅" if test_report["pytest"]["passed"] else "❌"
    flake8_icon = "✅" if test_report["flake8"]["passed"] else "⚠️"
    import_icon = "✅" if test_report["import_check"]["passed"] else "❌"

    report = f"""# 🌙 夜间自动化报告

> **运行时间**: {now}
> **运行环境**: Windows / Python {sys.version.split()[0]}

---

## 🧪 测试结果

| 检查项 | 状态 | 详情 |
|--------|------|------|
| Pytest | {pytest_icon} | {'通过' if test_report['pytest']['passed'] else '失败'} |
| Flake8 | {flake8_icon} | {'通过' if test_report['flake8']['passed'] else '有警告'} |
| 核心导入 | {import_icon} | {'全部通过' if test_report['import_check']['passed'] else '导入失败'} |

### Pytest 详情
```
{test_report['pytest']['detail'][-1500:]}
```

### Flake8 详情
```
{test_report['flake8']['detail'][-500:]}
```

---

## 📊 变更统计

- 最近 24h commits: **{len(changes.get('recent_commits', []))}**
- Aegis 架构变更: **{len(changes.get('track_aegis', []))}** 文件
- Agent 管理变更: **{len(changes.get('track_agent', []))}** 文件
- 面试材料变更: **{len(changes.get('track_interview', []))}** 文件

---

## 🐙 Github 子模块连线盘点

| 子项目 | 本地坐标 | 同步与仓库状态 |
|--------|----------|------|
"""
    if repo_status:
        for r in repo_status:
            report += f"| [{r['name']}]({r['url']}) | `{r['path']}` | {r['status']} |\n"
    else:
        report += "| (未测试) | - | - |\n"

    report += """
---

## 📝 更新的文件
- `cowokers_ai/CURRENT_TASK.md` — 三轨看板已更新
- `cowokers_ai/interview_changelog.md` — 面试材料已同步
- `logs/nightly/{datetime.now().strftime('%Y%m%d')}.log` — 详细日志

---

*此报告由 nightly_pipeline.py 自动生成*
"""

    REPORT_PATH.write_text(report, encoding="utf-8")

    # 同时写入日志文件
    log_file = LOGS_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_file.write_text(report, encoding="utf-8")

    print(f"  ✅ 报告已生成: {REPORT_PATH}")
    print(f"  📁 日志已保存: {log_file}")


# ============================================================
# 主入口
# ============================================================

def main():
    """主管线入口"""
    print("=" * 60)
    print(f"🌙 Aegis-Isle 夜间自动管线 v1.0")
    print(f"⏰ 启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    ensure_dirs()

    args = sys.argv[1:]

    if "--test-only" in args:
        test_report = phase_test_and_review()
        phase_generate_report(test_report, {})
    elif "--sync-resume" in args:
        changes = phase_update_dashboard()
        phase_sync_interview(changes)
    else:
        # 完整管线
        test_report = phase_test_and_review()
        changes = phase_update_dashboard()
        phase_sync_interview(changes)
        repo_status = phase_check_repos()
        phase_generate_report(test_report, changes, repo_status)

    print("\n" + "=" * 60)
    print("🎉 夜间管线执行完毕!")
    print("=" * 60)


if __name__ == "__main__":
    main()
