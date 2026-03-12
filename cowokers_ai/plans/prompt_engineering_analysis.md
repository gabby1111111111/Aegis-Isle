# CharLifeAgent Prompt 工程设计指南（基于 ST 预设分析）

通过分析提供的 SillyTavern 预设（【MoM】极光星河），我们提取了多项高级提示词（Prompt Engineering）技巧。这些技巧将直接用于设计 **CharLifeAgent** 的内心独白/日记生成 Prompt，以确保生成的文本具有极高的文学质量、不 OOC（Out of Character），并充满生命力。

---

## 核心 Prompt 设计技巧提取

### 1. 结构化块级指令 (Block-Level Instructions)
预设中大量采用了类似 HTML/XML 标签（如 `<writing_rules>`, `<snow_rules>`, `<thinking>`）来严格分隔不同的规则区块。
*   **启示**：在给 CharLifeAgent 的 Prompt 中，我们不应该写成一段长文本，而是应该结构化：
    ```xml
    <Role_Definition>你是 [角色的名字]，性格是...</Role_Definition>
    <Memory_Context>你今天经历了...</Memory_Context>
    <Writing_Constraints>不要使用...</Writing_Constraints>
    ```

### 2. 强有力的否定式“禁令” (Negative Constraints)
比起告诉 LLM“应该怎么做”，ST 预设往往通过**绝对的禁令**来拔高文笔上限。例如预设中严格规定：
*   **字词禁令**：禁止滥用特定副词（如“似乎”、“仿佛”、“哪怕”）、禁止机械的并列和排比连词。
*   **修辞禁令**：要求“隐喻（Metaphor）使用量为 0”，强迫 LLM 写出直白的动作而非浮夸的比喻。
*   **行为禁令**：禁止写出“不是…而是…”这种说教式句型。
*   **启示**：CharLifeAgent 生成日记时，很容易写出“今天我感到非常开心，因为……”这种像小学生日记一样的废话文本。我们必须加入严苛的禁令：**禁止直接描写情绪词语（开心、悲伤等），禁止写空洞的总结感言。**

### 3. 严格践行“展示而非告知” (Show, Don't Tell)
预设中强调，所有情绪必须通过客观事物的刻画、微表情、肢体动作或呼吸频率来“展示”。
*   **启示**：当 LifeEventBus 传来“User 今天刷了 3 篇法考小红书笔记”的事件时。
    *   ❌ **告知型（错误）**：“看到你这么努力，我感到很欣慰。”
    *   ✅ **展示型（正确）**：“咖啡杯底压着那几页被翻到卷边的法考真题。我把台灯的亮度调低了两档，看着光晕落在对面那人揉着眉骨的指尖上。”

### 4. 强制思维链 (ECoT - Enforced Chain of Thought)
预设最精妙的一点在于使用了 `[incipere]` 和 `<thinking>` 强制开启思维链模式。LLM 被要求在输出正文前，必须先按步骤思考：
1. 回顾了哪些约束？
2. 打算怎么用“展示而非告知”表达？
3. 角色的性格在这里该如何反应？
*   **启示**：CharLifeAgent 发起 API 调用时，我们可以要求：
    ```markdown
    请严格按照以下格式输出：
    <thinking>
    1. 发生事件：...
    2. 基于人设，我的第一反应是：...
    3. 为了不违反禁令，我打算描写的细节是：...
    </thinking>
    <diary>
    （这里才是最终要存入 FAISS 的高质量段落）
    </diary>
    ```
    用这种方法能消耗更多 token 但换取极稳定的高质量生成。

### 5. 动态小剧场/格式化输出 (Dynamic Scenarios & Formats)
预设中包含了大量的“小剧场”模板（如盲盒、论坛、系统状态栏等），它让 LLM 在特定的框架下搞创作。
*   **启示**：CharLifeAgent 的日记不仅可以是普通的文本，还可以被要求以特定的“体裁”入库。根据不同的情况，Prompt 可以要求 LLM 生成：
    *   `[待办备忘录]`：一条简短的备忘（例如：“明天得提醒她别忘记法考报名”）。
    *   `[随手涂鸦]`：一段看似随意的感叹。
    *   `[严肃日记]`：一段长篇的反思。

---

## 针对 CharLifeAgent 的实战 Prompt 雏形

根据上述启示，设计如下 CharLifeAgent 专属幕后 Prompt 模板：

```text
你现在是 {char_name}，正在自己的精神世界里进行反思。
这不会发送给 User，这是你自己记录的私密日志/心理活动。

<identity_rules>
{char_persona_yaml}
</identity_rules>

<recent_events>
{event_bus_data}
</recent_events>

<writing_constraints>
1. 【展示而非告知】：绝对禁止使用“开心、难过、期待”等抽象情绪词汇。通过你周围环境的互动、你身体的细微动作、或你关注的具体物品来展示情绪。
2. 【零比喻】：禁止使用“像…一样”、“仿佛”。用最克制、最白描的语言直击本质。
3. 【禁止升华】：禁止在结尾总结感悟或说教，在动作或一个未尽的念头处戛然而止。
4. 【符合人设】：绝不可 OOC。你的语言风格必须完全契合你的核心性格。
</writing_constraints>

请在 <thinking> 标签内进行 3 步思考，然后在 <autonomous_memory> 标签内写下 100-200 字的高质量日志内容，此内容将被持久化存储。
```
