---
name: auto-review-loop
description: 夜间代码自动修复与审查循环 (The Sleep-Safe Auto Loop - Antigravity 原生版)
---

# 🌀 夜间自动审查与修复循环 (Auto-Review-Loop)

> 参考来源: onsails/cc review-loop + levnikolaevich/claude-code-skills + Karpathy autoresearch
> 适配环境: Antigravity IDE (无需外部 API 额度，纯本地终端驱动)
> 主人: Gabby 大人 (早上起来看到满地红色报错会气死的 INFP 型 boss)

---

## 🎯 设计哲学

**你不是外部 API 审查员。你是一个"左脚踩右脚"的自省 Agent。**

终端 (Terminal) 的报错输出就是你的"审查导师"。你的工作循环是：
```
修改代码 → 跑终端命令 → 看报错 → 自我修正 → 再跑 → 直到绿灯
```

不需要花钱调用 Codex/GPT 来审查。Antigravity Agent 自己具备阅读终端输出和自我反思的能力。

---

## 🛡️ 五重安全阀 (Safety Valves)

### 📎 安全阀 1: 结界隔离分支 (Branch Sandboxing)
```
绝对红线: 不允许在 main 分支上做任何修改！
```
工作开始前，你必须:
```bash
git stash                              # 保护 Gabby 大人未保存的灵感
git checkout -b auto-fix/nightly-XXX   # XXX = 今晚任务的简短描述
```
这样即使你把整个分支搞炸了，Gabby 大人早上只需要一个 `git branch -D` 就能毁尸灭迹。

### 📎 安全阀 2: MAX_ROUNDS = 3 (防止无限循环)
对于**同一个文件**的同一个问题，最多修 3 次。
- Round 1: 初次修复
- Round 2: 如果终端仍然报错，换一种方式修
- Round 3: 最后一次机会
- **第 3 次仍然失败** → 立刻放弃该文件，在报告中标注 `[🚨 需 Gabby 亲自定夺]`

### 📎 安全阀 3: 逐文件隔离修改 (Per-File Isolation)
> 灵感来源: onsails/cc 的 per-issue subagent 设计

**绝对不允许一次性改 20 个文件然后统一跑测试。** 这是最常见的夜间翻车原因。
正确做法:
1. 选中 1 个文件
2. 只修这个文件
3. 立刻跑 `flake8 这个文件路径` 或 `pytest tests/对应测试.py`
4. 终端绿灯 → 提交这个文件的改动 → 进入下一个文件
5. 终端红灯 → 进入 Round 2（最多 3 轮）

这样即使某个文件改崩了，也不会污染其他已经修好的文件。

### 📎 安全阀 4: Escalation 升级标签 (核心代码免碰金牌)
以下文件/目录属于**核心神经索**，修改前必须三思:
```
src/aegis_isle/api/routers/memory.py     # API 主网关
src/aegis_isle/rag/st_memory_manager.py  # RAG 核心检索
src/aegis_isle/rag/embedder.py           # 嵌入模型加载
frontend/interview_app.py                # 面试系统主干
```
如果在修复某个 lint 警告时，发现需要动到这些文件的**逻辑结构**（不是纯格式修改），立刻:
- 停止修改
- 在报告中打上 `[🚨 需 Gabby 亲自定夺 - 涉及核心模块 XXX]`
- 只在该行加一个 `# FIXME: <描述>` 注释，然后跳到下一个文件

### 📎 安全阀 5: 误报追踪 (False Positive Tracking)
> 灵感来源: onsails/cc 的 false positive tracking

Gabby 大人的代码里有很多**故意这么写的东西**:
- 面试系统里用 emoji 做变量装饰
- 某些 bare except 是为了吞掉 ST 连接超时的异常
- 某些 f-string 没有 placeholder 是历史遗留

如果你确认某个 lint 警告属于"项目特性而非 Bug"，在该行添加 `# noqa: EXXX` 放行注释，
并在报告中记录为 `[✅ 已确认为项目特性，已放行]`，以后的循环不再碰它。

---

## 🚀 执行工作流 (The Loop)

### Step 0: 阅读军规
先读 `cowokers_ai/NIGHT_SHIFT_RULES.md`（如果存在的话），确认今晚的禁止操作区。

### Step 1: 切出隔离分支
```bash
git stash
git checkout -b auto-fix/flake8-nightly-YYYYMMDD
```

### Step 2: 扫描靶标
```bash
flake8 src/ --max-line-length=120 --count --statistics --format=pylint
```
从输出中提取**报错最密集的前 10 个文件**，按问题数量从高到低排序。

### Step 3: 逐文件循环修复 (The Core Loop)

```
FOR 每个靶标文件 (最多处理 10 个文件):

    round = 0
    WHILE round < 3:
        round += 1

        1. 用编辑工具修复该文件中的 lint 问题
           - 优先修: F401(未使用导入), F841(未使用变量), W292(末尾换行)
           - 谨慎修: E722(bare except) → 确认是否为故意设计
           - 跳过:   涉及核心模块逻辑变更的

        2. 跑终端验证:
           flake8 <该文件路径> --max-line-length=120

        3. 判断:
           - 终端输出为空 (零问题) → ✅ 该文件修复成功，break
           - 终端仍有报错 → 阅读报错内容，进入下一轮
           - 报错涉及核心逻辑 → 打 Escalation 标签，break

    IF round == 3 AND 仍有报错:
        在报告中标注 [🚨 需 Gabby 亲自定夺]

    git add <该文件>
    git commit -m "auto-fix: cleaned lint issues in <文件名>"
```

### Step 4: 全局回归验证
所有文件处理完后，跑一次完整测试确保没有连锁炸弹:
```bash
python -m pytest tests/ -v --tb=short -q
```
- 全绿 → 继续
- 有红 → 检查是不是你刚才的修改导致的。如果是，revert 那个文件的改动。

### Step 5: 封存与汇报
```bash
git add .
git commit -m "auto-fix: Antigravity nightly loop completed - X files fixed, Y skipped"
```

生成战报文件 `cowokers_ai/LOOP_REPORT.md`:
```markdown
# 🌙 夜间自动修复战报

> 运行时间: YYYY-MM-DD HH:MM
> 分支: auto-fix/flake8-nightly-YYYYMMDD
> 总循环轮数: X

## ✅ 已修复的文件 (X 个)
| 文件 | 修复的问题 | 轮数 |
|------|-----------|------|
| src/xxx.py | F401, W292 | 1 |

## 🚨 需 Gabby 亲自定夺 (Y 个)
| 文件 | 原因 |
|------|------|
| src/yyy.py | 涉及 memory.py 核心逻辑 |

## ✅ 已确认为项目特性，已放行 (Z 个)
| 文件 | 行号 | 规则 | 理由 |
|------|------|------|------|
| src/zzz.py | L42 | E722 | 故意吞掉 ST 超时异常 |

## 📊 回归测试
- pytest: PASSED / FAILED (详情)
```

### Step 6: 休眠
停止所有操作。等待 Gabby 大人早上通过 `/morning` 触发晨间汇报。
Bubby 总管家会在汇报时提醒: "昨晚夜班小弟在 `auto-fix/xxx` 分支上有动作，请 Review。"

---

## ⚡ 召唤咒语模板 (复制粘贴给新 Agent)

Gabby 大人睡前在 Agent Manager 新建对话时，复制这段话丢给新 Agent:

```
我要睡觉了。请你立即阅读并激活以下技能文件:
e:\Aegis_Isle\AegisIsle_cc_ver\Aegis-Isle\.agent\skills\auto-review-loop\SKILL.md

严格按照里面的 5 重安全阀和逐文件循环流程执行。
今晚的任务: 清理 src/ 目录下 flake8 报错最多的前 10 个文件。
绝对不允许动 main 分支。
绝对不允许修改核心模块的逻辑。
完成后生成 LOOP_REPORT.md 战报。
晚安，别搞砸了。
```
