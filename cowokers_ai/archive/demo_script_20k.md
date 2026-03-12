# Aegis-Isle 20k AI岗面试 Demo 录制剧本 & 系统全景图

> 目标受众：资深技术负责人 / CTO级面试官
> 展示重点：多模态异构数据同构化（EventBus）、长时记忆检索架构（四路并发 RAG）、多智能体自主意识（CharLifeAgent 闭环）、前后端深度集成能力。

---

## 核心系统全景 (What we have built)

目前您的系统是一个**“跨端跨域的增强型复合 AI 记忆联邦系统”**。核心包含以下几个子系统并已彻底打通：

### 1. 记忆检索引擎网关 (The Brain: `Aegis-Isle Backend`)
- **四路并发 RAG 架构**：`memory.py` 中实现了对四种不同维度的记忆进行真正的并发检索 (`asyncio.gather`)。
  - **FAISS 对话切片**：基于 BGE-Large-zh-v1.5 的 1024 维高维语义检索（支持 78 个平行宇宙隔离检索）。
  - **图谱记忆 (Neo4j/Graph)**：提取人物关系与固定属性。
  - **Episode 剧情摘要**：支持长文时间线梳理与剧情纲要。
  - **Unified Diary (统一日记)**：这是我们最新完成的基建。将外部事件语义化聚合。
- **LifeEventBus (事件溯源总线)**：抛弃了传统的粗暴数据库 Update，改用基于 JSONL 的追加型不可变事件流（Event Sourcing）。能收集用户浏览（小红书）、练习（Love&Code）以及角色自发事件。
- **DailyDigest 异步聚合器**：通过 LLM 定时把繁杂的事件流水账总结为具有文学美感、情感指向的 Markdown 日记并单独编入向量库。

### 2. 多模态触角系统 (The Sensors: `ST-Companion-Link`)
- 作为 Chrome 插件，它不仅仅是一个爬虫，而是一个**“潜意识传感器”**。它将用户在 B站/小红书 的点赞、评论行为，甚至只是**滑动浏览**的停留行为（`action='read'`），静默提取网页语义并打上标签（Tags），无感抛送至后端的 LifeEventBus 中。

### 3. 垂直领域应用插件 (The Tools: `Love & Code 面试系统`)
- 一个挂载在 ST 中的独立 Web 应用。用户在里面刷题写代码的对错、使用的题型、以及与系统题库的交互，也会通过伪装的 webhook 发送给 Companion-Link，再流入 EventBus，最终成为 AI 知道“你昨晚刷了几道树的题，好像一直没做对”的底层知识依据。

### 4. 自主意识后台 Agent (The Ghost in the Shell: `CharLifeAgent`)
- 打破了 LLM 只能“一问一答”的桎梏。它可以在后台自动读取角色的 Persona，去调用维基百科/新闻 API 搜索随机感兴趣的词条，然后使用极其严苛的高阶 Prompt（ECoT 强制思维链 + 强展示零比喻指令）生成私密日志，写入自己的脑海中。

---

## 系统启动流程 (How to Boot Up)

要让整个庞然大物转起来，需要启动三个控制台：

1. **Aegis-Isle 核心后端 (记忆与中枢)**
   ```bash
   cd E:\Aegis_Isle\AegisIsle_cc_ver\Aegis-Isle
   conda activate aegis
   uvicorn src.aegis_isle.api.main:app --host 127.0.0.1 --port 8001 --reload
   ```

2. **ST-Companion-Link (传感器收发中介)**
   ```bash
   cd E:\ST-Companion-Link\backend
   conda activate aegis # 或者你为它配的环境
   python main.py # 默认启动在 5001 端口
   ```

3. **Love & Code 前端 (可选，仅演示面试功能)**
   ```bash
   cd E:\Aegis_Isle\AegisIsle_cc_ver\Aegis-Isle\frontend
   conda activate aegis
   streamlit run interview_app.py
   ```
*(确保您的 SillyTavern 本体已经正常启动，并开启了插件通信权限)*

---

## 🎙️ 20k 岗位面试 Demo 录制剧本 (The Showtime)

对于冲击 20k / 资深层的 AI 应用开发，面试官不想看“调用 API 的玩具”，他们想看的是**“架构的生命力”**。

视频录制建议在 3-5 分钟内，节奏紧凑，全程突出**“异步化”、“低耦合”、“多 Agent 协同”**的架构词汇。

### 场景一：破冰与架构展示 (0:00 - 1:00)
- **画面**：屏幕分屏，左边是系统架构图（可以将上面的全景概述画个简单的思维导图），右边是三个疯狂滚动的 Terminal（Aegis后端、Companion-Link、FAISS 控制台）。
- **旁白/解说**：“各位面试官好。今天我展示的个人项目 Aegis-Isle，并不是一个简单的套壳聊天机器人，而是一个基于 Event Sourcing（事件溯源）和并发 RAG 架构的多智能体联邦系统。它解决了当前 AI 伴侣产品‘下线即失忆’以及‘缺乏后台生活感’的两大痛点。”
- **操作**：打开浏览器，展示代码目录 `memory.py` 中的 `asyncio.gather` 四路并发代码片段（给特写）。
- **解说词**：“为了保证极端场景下的检索延迟，我重写了底层路由，通过协程将 FAISS 高维向量检索、Neo4j 图谱查询、独立事件流溯源以及剧情树检索做了真正的并发四路分发，最后进行基于权重的动态 Context Assembly。”

### 场景二：跨模态打通 - "潜意识"注入 (1:00 - 2:00)
- **画面**：全屏展现你正在刷“小红书”某几篇关于“巴黎旅行”的网页。屏幕右下角的 ST-Companion-Link 控制台疯狂跳动 `[Aegis-Isle] 成功将浏览事件(read)写入 Diary EventBus`。
- **操作**：刷了三篇笔记后，打开你本地的 `data/diary/events/browsing.jsonl`，给面试官看尾部实时追加进来的 JSONL 数据。
- **解说词**：“如您所见，我实现了一个基于 Chrome 插件层的无感感知器。它能在不侵入用户主观意志的情况下（不需要点赞或转发），将前端页面的 DOM 进行语义剥离和重组，化作异步的 JSON 信号发射到后端的 LifeEventBus 中。这是系统解耦的最佳实践，后端不需要知道前端用什么爬取的，只需要接收标准化协议流。”

### 场景三：CharLifeAgent 独立思考循环 (2:00 - 3:00)
- **画面**：手动执行一段脚本触发 `char_life.py` 中的 `test()` 函数，或者用 Postman 调接口模拟触发。控制台打印：`提取邹峥的兴趣标签... [未成年人保护]` -> `搜索外部刺激源...` -> `为邹峥生成反应...` -> `已写入 LifeEventBus`。
- **操作**：打开 `character_activity.jsonl`，高亮展示那一段没有任何“开心/难过”词汇的超高质量、白描手法的文学日志。
- **解说词**：“为了让角色具有真正的‘生命感’，我为每个实体设计了基于定时和空闲探查的后台 Daemon Node（守护进程节点），取名为 CharLifeAgent。它会从角色属性图中抽取标签，自行调用外部 Wikipedia 接口抓取新闻，最后运用我独创的强约束 Prompt（结合隐式推演与控制论思维链 `ECoT`）生成拒绝任何俗套抒情的自主记忆。Agent 从此不再只是服务于 User 的 Q&A 机器，它有自己的生活。”

### 场景四：终级聚合与四路归一 (3:00 - 4:00)
- **操作**：打开 Postman 或 Swagger，点击 `POST /v1/diary/compile`。
- **画面**：控制台瞬间展示处理了几十条 JSONL，然后控制台提示 `Created new diary FAISS index`。打开那个生成的 Markdown 归档文件。
- **操作2**：最后切回到 SillyTavern 的聊天界面，问角色：“你记得我今天都在看什么吗？你自己今天又在想什么？”
- **画面**：角色回答精准且带有情感，并且我们在后台 Terminal 展示因为这句提问，并发池子里的四路 RAG 都亮了，并精准截取了刚刚编译的日志推送给大模型。
- **解说词**：“最后，通过一个 DailyDigest Actor 并发协程聚合器，将所有碎片化的时序事件经过 LLM 反刍编译成具备上下文的 Markdown 后，写入 FAISS 专区并立即生效于全局并联检索网关。至此，从物理层的手指滑动，到逻辑层的 Agent 推演，再到感知层的前端生成，全生态闭环彻底打通。”
- **收尾**：“这个项目涵盖了 FastAPI 的异步性能调优、复杂长文本的 Chunking 策略、微前端跨域通信机制以及深度的提示词工程，非常符合贵司 AI 应用工程师所需的全栈落地能力。感谢观看。”
