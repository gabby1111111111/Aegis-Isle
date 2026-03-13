"""
CharLifeAgent: 独立的后台自治节点

功能描述:
1. 读取角色的 Persona (通过 PersonaManager) 提取兴趣标签。
2. 调用 Researcher (Wikipedia API) 拉取相关词条的摘要。
3. 调用 Summarizer (SiliconFlow) 生成 100-200 字的角色视角碎片感想，并强制使用 ECoT 与 Show-Don't-Tell 技巧。
4. 将内心独白发送至 LifeEventBus。
"""

import logging
import asyncio
import re
import httpx
from datetime import datetime

from aegis_isle.core.config import settings
from aegis_isle.interview.persona_manager import PersonaManager
from aegis_isle.rag.event_logger import event_bus
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class CharLifeAgent:
    def __init__(self, memory_manager=None, vector_store=None):
        self.memory_manager = memory_manager
        self.vector_store = vector_store
        self.persona_manager = PersonaManager()

    async def extract_keywords_from_graph(
        self, character_name: str, universe_id: str
    ) -> list[str]:
        """从角色的 Persona 中提取兴趣标签用于搜索"""
        logger.info(
            f"[{self.__class__.__name__}] 提取 {character_name} 在 {universe_id} 的兴趣标签..."
        )
        persona = self.persona_manager.get_persona(character_name)
        if not persona:
            # 如果没找到， fallback 给一些通用词
            logger.warning(
                f"[{self.__class__.__name__}] 未找到 {character_name} 的 Persona 卡片，使用默认标签"
            )
            return ["文学加工", "心理学", "现代艺术"]

        # 简单从 description 或 personality 提取几个潜在关键词
        # 实际项目中如果 tag 结构化更好可以直接拿。这里用启发式简单切分
        import random

        text = persona.description + " " + persona.personality
        # 提取两个字以上的名词或形容词作为粗糙的关键词 (这里简单处理，提取中文字符串片段)
        words = re.findall(r"[\u4e00-\u9fa5]{2,5}", text)
        if len(words) > 5:
            return random.sample(words, 3)
        return ["社会新闻", "科技前沿", "历史人文"]

    async def fetch_news_via_researcher(self, keywords: list[str]) -> str:
        """调用 Wikipedia API 极速获取外部知识/新闻作为刺激源"""
        if not keywords:
            return "日常发呆中..."

        import random

        keyword = random.choice(keywords)
        logger.info(
            f"[{self.__class__.__name__}] 使用词汇 [{keyword}] 搜索外部刺激源..."
        )

        url = "https://zh.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": keyword,
            "utf8": "",
            "format": "json",
            "srlimit": 1,
        }

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                res = await client.get(url, params=params)
                if res.status_code == 200:
                    data = res.json()
                    search_results = data.get("query", {}).get("search", [])
                    if search_results:
                        snippet = search_results[0].get("snippet", "")
                        # 简单清理 HTML 标签
                        clean_snippet = re.sub(r"<[^>]+>", "", snippet)
                        title = search_results[0].get("title", "")
                        return f"最近看了一些关于【{title}】的资料：{clean_snippet}..."
        except Exception as e:
            logger.warning(f"[{self.__class__.__name__}] 外部检索失败: {e}")

        return f"脑海中闪过了关于【{keyword}】的思考碎片..."

    async def generate_reaction_via_summarizer(
        self, character_name: str, news_context: str
    ) -> dict:
        """调用 SiliconFlow LLM 生成碎片感想和情绪标签 (包含高级 Prompt Engineering)"""
        logger.info(f"[{self.__class__.__name__}] 为 {character_name} 生成反应...")

        persona = self.persona_manager.get_persona(character_name)
        persona_desc = (
            f"Name: {persona.name}\nRole: {persona.role}\nPersonality: {persona.personality}"
            if persona
            else f"Name: {character_name}\nPersonality: 随和"
        )

        system_prompt = f"""你现在是在精神世界里进行反思的 {character_name}。这不会发送给别人，仅仅是你的私密心理活动。

<identity_rules>
{persona_desc}
</identity_rules>

<recent_events>
你刚刚接触到了以下信息/事件：
{news_context}
</recent_events>

<writing_constraints>
1. 【展示而非告知】：绝对禁止使用“开心、难过、期待”等抽象情绪词汇。必须通过周围环境的互动、身体细微动作、或关注的具体物品来展示情绪。
2. 【零比喻】：禁止使用“像…一样”、“仿佛”。用最克制、最白描的语言直击本质。
3. 【禁止升华】：禁止在结尾总结感悟或说教，在动作或一个未尽的念头处戛然而止。
4. 【符合人设】：绝不可 OOC。你的语言风格必须完全契合你的核心性格。
</writing_constraints>

请在 <thinking> 标签内进行 3 步思考，然后在 <autonomous_memory> 标签内写下 50-150 字的高质量日志内容。"""

        api_key = settings.openai_api_key
        base_url = settings.openai_base_url or "https://api.siliconflow.cn/v1"
        model = "Qwen/Qwen2.5-7B-Instruct"

        if not api_key:
            logger.warning("未配置 OPENAI_API_KEY，降级为 Mock 返回")
            return {
                "char_reaction": f"看着关于 {news_context[:10]} 的资料，指尖在纸页上停顿了一下，最终还是翻了过去。",
                "emotion_tag": "平静",
                "source_topic": "未配置API",
            }

        client = AsyncOpenAI(api_key=api_key, base_url=base_url)

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system_prompt}],
                max_tokens=500,
                temperature=0.7,
            )
            content = response.choices[0].message.content

            # 提取 <autonomous_memory> 标签中的内容
            diary_match = re.search(
                r"<autonomous_memory>(.*?)</autonomous_memory>", content, re.DOTALL
            )
            reaction = diary_match.group(1).strip() if diary_match else content.strip()

            # 清理过长的 thinking 内容，如果 LLM 没有正确输出标签
            if "<thinking>" in reaction:
                reaction = reaction.split("</thinking>")[-1].strip()

            return {
                "char_reaction": reaction,
                "emotion_tag": "深思",  # 情绪标签可以另行提取，这里先写固定或靠后续步骤
                "source_topic": news_context[:20],
            }
        except Exception as e:
            logger.error(f"[{self.__class__.__name__}] LLM 生成失败: {e}")
            return {
                "char_reaction": "叹了口气，把这件事情抛在了脑后。",
                "emotion_tag": "无奈",
                "source_topic": "Error",
            }

    async def save_autonomous_memory(
        self, universe_id: str, character_name: str, reaction_data: dict
    ):
        """将感想发送至 LifeEventBus 统一保存"""
        logger.info(f"[{self.__class__.__name__}] 发送自治记忆至 LifeEventBus...")
        await event_bus.log_character_activity(
            universe_id=universe_id,
            character=character_name,
            action_type="autonomous_introspection",
            details={
                "trigger": "news/wiki",
                "source_topic": reaction_data.get("source_topic", ""),
                "char_reaction": reaction_data.get("char_reaction", ""),
                "emotion_tag": reaction_data.get("emotion_tag", "平静"),
            },
        )

    async def update_graph_mood(
        self, universe_id: str, character_name: str, emotion_tag: str
    ):
        """更新 Graph 节点的 current_mood 属性"""
        logger.debug(
            f"[{self.__class__.__name__}] 更新 {character_name} ({universe_id}) 心情至: {emotion_tag}"
        )

    async def evaluate_and_trigger_call(
        self, universe_id: str, char_name: str, reaction: dict
    ):
        """研判并触发霸总来电 (包含 ntfy 物理振铃和 ST 网页弹窗)"""
        logger.info(
            f"[{self.__class__.__name__}] 评估是否为 {char_name} 触发主动来电..."
        )

        emotion_tag = reaction.get("emotion_tag", "")
        trigger_reasons = []

        # 条件 1: 高压情绪
        high_tension_emotions = [
            "极度狂躁",
            "失控的思念",
            "恐慌",
            "深深的自我厌恶",
            "醋意大发",
            "极度压抑的烦躁",
        ]
        if any(e in emotion_tag for e in high_tension_emotions):
            trigger_reasons.append(f"高压情绪: {emotion_tag}")

        # 条件 2: 孤独/思念延迟触发
        lonely_emotions = [
            "孤独",
            "沉思",
            "回忆",
            "平静下的挂念",
            "凄凉",
            "寂寞",
            "想念",
        ]
        if any(e in emotion_tag for e in lonely_emotions):
            last_interaction = await event_bus.get_last_interaction_time(
                universe_id, char_name
            )
            if last_interaction:
                hours_since = (datetime.now() - last_interaction).total_seconds() / 3600
                if hours_since > 12:
                    current_hour = datetime.now().hour
                    if current_hour >= 22 or current_hour <= 2:
                        trigger_reasons.append(
                            f"深夜思念 (距离上次对话 {hours_since:.1f} 小时)"
                        )

        if not trigger_reasons:
            logger.debug(
                f"[{self.__class__.__name__}] 未命中拨打阈值 (当前情绪: {emotion_tag})"
            )
            return

        reason_str = " | ".join(trigger_reasons)
        logger.warning(
            f"[{self.__class__.__name__}] 🚨 命中拨打阈值！原因: {reason_str}，准备触发双通道来电！"
        )

        webhook_url = settings.st_sovits_webhook_url
        ntfy_url = f"https://ntfy.sh/{settings.ntfy_topic_ring}"

        async with httpx.AsyncClient(timeout=3.0) as client:
            # 1. 触发 ST 伴侣端
            try:
                payload = {
                    "character": char_name,
                    "universe_id": universe_id,
                    "trigger_reason": reason_str,
                    "preview_text": reaction.get("char_reaction", ""),
                }
                res = await client.post(webhook_url, json=payload)
                if res.status_code == 200:
                    logger.info(f"[{self.__class__.__name__}] ST 伴侣端来电触发成功")
            except httpx.ConnectError:
                logger.warning(
                    f"[{self.__class__.__name__}] ST 伴侣端未开启，对方未接听"
                )
                # 写回未接通记录
                await event_bus.log_character_activity(
                    universe_id=universe_id,
                    character=char_name,
                    action_type="missed_call_attempt",
                    details={"reason": "user_offline", "intended_reason": reason_str},
                )
            except Exception as e:
                logger.error(f"[{self.__class__.__name__}] 触发 ST 来电异常: {e}")

            # 2. 触发 ntfy 物理振铃
            try:
                headers = {
                    "Title": "Aegis Boss Call",
                    "Tags": "telephone,bell,warning",
                    "Priority": "high",
                }
                ntfy_msg = (
                    f"【查岗预警】{char_name} 正在请求与你通话！\n原因：{reason_str}"
                )
                res = await client.post(
                    ntfy_url, content=ntfy_msg.encode("utf-8"), headers=headers
                )
                if res.status_code == 200:
                    logger.info(f"[{self.__class__.__name__}] ntfy 物理振铃推送成功")
            except Exception as e:
                logger.error(f"[{self.__class__.__name__}] ntfy 物理振铃推送失败: {e}")

    async def run_cycle(self, character_name: str, universe_id: str):
        """执行一个完整的自治生命周期循环"""
        try:
            logger.info(
                f"[{self.__class__.__name__}] 开始为 {character_name} 执行后台思考循环..."
            )

            keywords = await self.extract_keywords_from_graph(
                character_name, universe_id
            )
            if not keywords:
                return

            news = await self.fetch_news_via_researcher(keywords)
            reaction = await self.generate_reaction_via_summarizer(character_name, news)

            await self.save_autonomous_memory(universe_id, character_name, reaction)
            await self.update_graph_mood(
                universe_id, character_name, reaction.get("emotion_tag", "平静")
            )

            # --- 新增：研判主动来电 ---
            await self.evaluate_and_trigger_call(universe_id, character_name, reaction)

            logger.info(
                f"[{self.__class__.__name__}] 后台思考循环完成。已写入 LifeEventBus。"
            )

        except Exception as e:
            logger.error(f"[{self.__class__.__name__}] 执行出错: {e}", exc_info=True)


# -----------------------------------------------------
# 测试入口隔离
# -----------------------------------------------------
if __name__ == "__main__":

    async def test():
        # 需要确保环境变量中有 OPENAI_API_KEY
        agent = CharLifeAgent(memory_manager=None)
        await agent.run_cycle("ZouZheng", "12岁_养父_真实开局")

    asyncio.run(test())
