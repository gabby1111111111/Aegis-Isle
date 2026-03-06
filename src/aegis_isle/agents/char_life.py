"""
CharLifeAgent: 独立的后台自治节点

功能描述:
1. 定时或对话间隙触发。
2. 读取角色的 `graph_nodes` 中的 `tags`/`hobbies`/`occupation` 构建搜索关键词。
3. 调用 Researcher 拉取相关新闻/学术动态。
4. 调用 Summarizer 生成 100-200 字的角色视角碎片感想。
5. 存入 FAISS，类型标记为 `autonomous_memory`。
6. 更新 Graph 里该角色节点的 `current_mood` 属性。
"""

import json
import logging
import asyncio
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

class CharLifeAgent:
    def __init__(self, memory_manager, vector_store=None):
        self.memory_manager = memory_manager
        self.vector_store = vector_store

    async def extract_keywords_from_graph(self, character_name: str, universe_id: str) -> list[str]:
        """从 Graph nodes 提取搜索关键词"""
        # TODO: 从 neo4j / 真实 graph 存储中读取该角色的 attributes
        # 模拟返回
        logger.info(f"[{self.__class__.__name__}] 提取 {character_name} 在 {universe_id} 的兴趣标签...")
        return ["刑法学动态", "未成年人保护", "古典音乐"]

    async def fetch_news_via_researcher(self, keywords: list[str]) -> str:
        """调用 Researcher 节点获取外部新闻"""
        # TODO: 集成 Tavily / 真实 Researcher 逻辑
        logger.info(f"[{self.__class__.__name__}] 搜索新闻: {keywords}")
        return "今日头条：未成年人保护法修订草案公开征求意见，涉及监护人责任认定标准的变化。"

    async def generate_reaction_via_summarizer(self, character_name: str, news_context: str) -> dict:
        """调用 Summarizer 生成碎片感想和情绪标签"""
        # TODO: 注入 Prompt 调用 LLM 生成 100-200 字感想
        logger.info(f"[{self.__class__.__name__}] 为 {character_name} 生成反应...")
        return {
            "char_reaction": "让他想起上周课上那个学生的提问，关于监护权边界的探讨总是那么苍白...",
            "emotion_tag": "沉思/轻微烦躁",
            "source_topic": "未成年人保护法修订"
        }

    async def save_autonomous_memory(self, universe_id: str, character_name: str, reaction_data: dict):
        """将感想存入 FAISS 并标记类型为 autonomous_memory"""
        # TODO: 封装为 ChatChunk (或 Document) 存入 FAISS
        memory_doc = {
            "type": "autonomous_memory",
            "universe_id": universe_id,
            "char": character_name,
            "trigger": "news",
            "source_topic": reaction_data.get("source_topic", ""),
            "char_reaction": reaction_data.get("char_reaction", ""),
            "emotion_tag": reaction_data.get("emotion_tag", "平静"),
            "timestamp": datetime.now().isoformat()
        }
        logger.info(f"[{self.__class__.__name__}] 保存自治记忆: {json.dumps(memory_doc, ensure_ascii=False)}")
        # await self.memory_manager.ingest_chunks(...)

    async def update_graph_mood(self, universe_id: str, character_name: str, emotion_tag: str):
        """更新 Graph 节点的 current_mood 属性"""
        # TODO: 执行 Graph DB 更新 (如 neo4j Cypher)
        logger.info(f"[{self.__class__.__name__}] 更新 {character_name} ({universe_id}) 心情至: {emotion_tag}")

    async def run_cycle(self, character_name: str, universe_id: str):
        """执行一个完整的自治生命周期循环"""
        try:
            logger.info(f"[{self.__class__.__name__}] 开始为 {character_name} 执行后台思考循环...")
            
            keywords = await self.extract_keywords_from_graph(character_name, universe_id)
            if not keywords:
                return

            news = await self.fetch_news_via_researcher(keywords)
            reaction = await self.generate_reaction_via_summarizer(character_name, news)
            
            await self.save_autonomous_memory(universe_id, character_name, reaction)
            await self.update_graph_mood(universe_id, character_name, reaction.get("emotion_tag", "平静"))
            
            logger.info(f"[{self.__class__.__name__}] 后台思考循环完成。")
            
        except Exception as e:
            # 必须隔离报错，避免影响主系统
            logger.error(f"[{self.__class__.__name__}] 执行出错: {e}", exc_info=True)


# -----------------------------------------------------
# 测试入口隔离
# -----------------------------------------------------
if __name__ == "__main__":
    async def test():
        agent = CharLifeAgent(memory_manager=None)
        await agent.run_cycle("邹峥", "12岁_养父_真实开局")
        
    asyncio.run(test())
