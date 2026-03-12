# Love & Code 系统接入统一日记池与潜意识 Buffer 改造计划

本方案旨在零侵入地利用 ST-Companion-Link 已有的 15 分钟短期记忆机制（Read Buffer）以及事件分发机制，让《Love & Code》面试系统能够完美连入酒馆角色的大脑。

## 设计思路

我们将《Love & Code》的 Python 前端（`interview_app.py`）伪装成一个小红书的浏览器扩展插件，直接通过 HTTP 异步请求向 ST-Companion-Link 的 5001 端口发送 `SignalPayload` 事件。

这样设计的好处是：
1. **完全不需要修改 ST-Companion-Link 的底层代码**。
2. ST-Companion-Link 会自动帮我们把“刷题记录”加入 15 分钟的潜意识上下文。
3. ST-Companion-Link 会自动通过 Webhook 把事件落入未来的 Aegis `LifeEventBus`（统一日记池）。

---

## 改造详情与代码参考

我们需要修改 `E:\Aegis_Isle\AegisIsle_cc_ver\Aegis-Isle\frontend\interview_app.py` 中的关键函数，加入发送信号的逻辑。

### 1. 挂载发包函数 (基础建设)
在文件顶部或全局区域新增异步发包函数：

```python
import httpx
import logging

logger = logging.getLogger("love_and_code.tracker")

async def send_to_companion_link(action: str, title: str, tags: list, comment_text: str = None):
    """
    伪装浏览器扩展，向 ST-Companion-Link 发送信号
    """
    url = "http://localhost:5001/api/signal"
    payload = {
        "action": action,
        "note_url": "love_and_code://interview", # 虚构伪协议
        "comment_text": comment_text,
        "note_data": {
            "title": f"[Love & Code面试练习] {title}",
            "tags": tags + ["面试练习", "Love&Code"],
            "platform": "love_and_code",
            "author": {"nickname": "系统题库"}
        }
    }
    try:
        async with httpx.AsyncClient() as client:
            await client.post(url, json=payload, timeout=2.0)
            logger.info(f"✅ 已发送 {action} 信号至 Companion-Link: {title}")
    except Exception as e:
        logger.warning(f"⚠️ 发送 Companion-Link 信号失败 (服务未开启?): {e}")
```

### 2. 改造点 A：生成新题时送入 15 分钟潜意识 (Read)
定位函数：`generate_new_question()`
当用户抽取到一道新题时，作为一次**阅读 (read)** 行为发送。

```python
# 在 st.session_state.current_question = question 后触发：
import asyncio
question_title = question.question_text[:30] + "..." # 取题目开头作为标题
extracted_tags = [] # 如果题目有分类字段，例如 question.review_box 等

# 异步触发（Background），不阻塞 Streamlit 渲染
asyncio.create_task(send_to_companion_link(
    action="read",
    title=question_title,
    tags=extracted_tags
))
```

### 3. 改造点 B：提交作答时瞬间引爆主动谈话 (Comment)
定位函数：`submit_answer()`
当用户完成一套答题逻辑时，发送一次带感情色彩的 **交互 (comment/like)** 行为以触发角色发言。

```python
# 在判断 is_correct 后：
reaction_text = f"刚才做对了一道面试题，得分+15！好耶！" if is_correct else f"这题又答错了，好郁闷，被扣了10分..."
if is_partial:
    reaction_text = "这道面试题答得磕磕绊绊，只拿到一部分分数..."

# 异步发送强互动信号
asyncio.create_task(send_to_companion_link(
    action="comment",
    title="本次答题结果结算",
    tags=["答题反馈"],
    comment_text=reaction_text
))
```

---

## 简单测试举例与预期链路

1. **环境准备**：
   - 启动 SillyTavern 并打开预设角色。
   - 启动 `ST-Companion-Link` 后端（端口 5001）。
   - 启动《Love & Code》系统。

2. **行为 1：疯狂刷题（Read 累积）**：
   - 在面试系统里连翻 3 道题目（但不提交）。
   - **后端现象**：此时 `5001` 后端默默收到 3 次 `action="read"`，`read_buffer.py` 内的 `deque` 里存好了这 3 道题目的摘要（包含 "面试练习", "Love&Code" 等 tag）。SillyTavern 终端并不会让角色说话。

3. **行为 2：提交答案（Comment 爆发）**：
   - 认真做完第 3 题，点击提交。系统判对，发送 `action="comment"`。
   - **后端现象**：`5001` 后端收到 comment，立即提取刚才 15 分钟的 3 道题缓存摘要，打包这句“刚做对了一道面试题！”，发送给 SillyTavern。
   - **SillyTavern 现象**：角色突然主动开口说话：“看你刚才这好一会儿一直在埋头刷 Love & Code 的面试题，这回总算全对了吧？太棒啦，想要什么奖励？”

4. **长远结果**：
   - 这 4 次事件也会通过 `dispatcher.py` 中的 Webhook 被静默转发到尚未完成的 `LifeEventBus`（Aegis 8001 端口），成为 `browsing.jsonl` 中的 4 行永久日志记录！
