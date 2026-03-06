"""
Aegis-Isle 长线记忆 API 路由

提供两个核心接口供 SillyTavern 插件调用：
1. POST /v1/memory/search  - 查询与当前消息相关的历史记忆片段
2. POST /v1/memory/ingest  - 将新的对话片段存入记忆库
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import logging

from ...rag.st_memory_manager import memory_manager
from ...rag.graph_searcher import graph_searcher
from ...rag.episode_searcher import episode_searcher
import asyncio
import os
import time

router = APIRouter()
logger = logging.getLogger(__name__)


# ============================================
# 请求/响应模型
# ============================================

class MemorySearchRequest(BaseModel):
    query: str                          # 用户刚刚发送的消息，用来做向量搜索
    character_name: str                 # 角色名，用于隔离不同角色的记忆库
    world_line: Optional[str] = None    # 可选：世界线标签（用于跨聊天同设定分组）
    k: int = 3                         # 返回的记忆片段数量，默认 3 段


class MemoryIngestRequest(BaseModel):
    character_name: str
    world_line: Optional[str] = None
    messages: List[dict]                # [{"role": "user", "name": "gabby", "content": "..."}, ...]
    chat_file: Optional[str] = "realtime"


class MemorySearchResponse(BaseModel):
    memories: List[dict]
    context_string: str                 # 格式化好的、可以直接塞进 system prompt 的字符串
    count: int
    debug_info: Optional[dict] = None   # 包含了三路路由是否命中、各路提取长度等元数据


# ============================================
# 接口：查询记忆
# ============================================

@router.post("/memory/search", response_model=MemorySearchResponse)
async def search_memory(req: MemorySearchRequest):
    """
    根据当前消息，从角色的记忆库里检索最相关的历史对话片段。
    
    ST 插件在用户发消息前调用这个接口，把返回的 context_string 注入 system prompt。
    
    示例:
        POST /v1/memory/search
        {
            "query": "你还记得那次在法餐厅的事吗？",
            "character_name": "ZouZheng",
            "world_line": "AIDom",
            "k": 3
        }
    """
    logger.info(f"[Memory] 查询 {req.character_name} 的记忆, query='{req.query[:30]}...'")
    
    try:
        query_text = req.query.lower()
        
        # 定义三路意图路由标记 (极速纯字符运算 <1ms)
        do_faiss = False
        do_graph = False
        do_episode = False
        
        if any(k in query_text for k in ["那段", "那时候", "当时", "气氛", "氛围", "记得", "说起"]):
            do_faiss = True
        if any(k in query_text for k in ["关系", "感觉", "对我", "喜欢", "什么样"]):
            do_graph = True
        if any(k in query_text for k in ["第一次", "什么时候", "发生过", "以前", "之前"]):
            do_episode = True
            
        # Fallback：没命中任何特定意图的话，全开 FAISS 和 Episode 以防遗漏
        if not do_faiss and not do_graph and not do_episode:
            do_faiss = True
            do_episode = True
            
        # 定义协程包
        async def _run_faiss():
            if not do_faiss: return []
            docs = await memory_manager.search_memory(req.query, req.character_name, req.world_line, req.k)
            return docs
            
        async def _run_graph():
            if not do_graph: return ""
            return await graph_searcher.search(req.query, req.world_line or "", req.character_name)
            
        async def _run_episode():
            if not do_episode: return ""
            return await episode_searcher.search(req.query, req.world_line or "")

        # 并发执行三路检索
        docs, graph_text, episode_text = await asyncio.gather(
            _run_faiss(),
            _run_graph(),
            _run_episode()
        )
        
        # 结果拼装
        faiss_text = memory_manager.format_context_for_prompt(docs) if docs else ""
        
        final_context_parts = []
        if graph_text: final_context_parts.append(graph_text)
        if episode_text: final_context_parts.append(episode_text)
        if faiss_text: final_context_parts.append(f"【详细记忆】\n{faiss_text}")
        
        final_context_string = "\n\n".join(final_context_parts)
        
        memories = []
        for doc in docs:
            memories.append({
                "text": doc.page_content,
                "chat_file": doc.metadata.get("chat_file", "unknown"),
                "start_time": doc.metadata.get("start_time"),
            })
            
        # 把路由命中信息一并返回给前端，方便 index.js 控制台打印
        debug_info = {
            "routed_faiss": do_faiss,
            "routed_graph": do_graph,
            "routed_episode": do_episode,
            "faiss_len": len(faiss_text),
            "graph_len": len(graph_text),
            "episode_len": len(episode_text)
        }
        
        logger.info(f"[Memory] 并发查询完成, context总长度: {len(final_context_string)}")
        
        return MemorySearchResponse(
            memories=memories,
            context_string=final_context_string,
            count=len(docs),
            debug_info=debug_info
        )
        
    except Exception as e:
        logger.error(f"[Memory] 查询记忆失败: {e}", exc_info=True)
        return MemorySearchResponse(memories=[], context_string="", count=0)

# ============================================
# 接口：接收前端保存完整 Prompt 用于 Debug
# ============================================

class DebugSaveRequest(BaseModel):
    universe_id: str
    prompt_text: str

@router.post("/memory/debug_save")
async def debug_save_prompt(req: DebugSaveRequest):
    """供前端调用的 Debug 保存接口，将最终拼接好的 Prompt 持久化"""
    try:
        if os.environ.get("DEBUG_SAVE", "").lower() != "true":
            return JSONResponse({"status": "skipped", "reason": "DEBUG_SAVE not enabled"})
            
        debug_dir = os.path.join(os.getcwd(), "debug", "prompts")
        os.makedirs(debug_dir, exist_ok=True)
        
        safe_world = "".join([c for c in req.universe_id if c.isalnum() or c in (' ', '-', '_')]).strip()
        timestamp = int(time.time() * 1000)
        file_path = os.path.join(debug_dir, f"prompt_{safe_world}_{timestamp}.txt")
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(req.prompt_text)
            
        logger.info(f"[Memory] 完整 Prompt 已保存至: {file_path}")
        return JSONResponse({"status": "saved", "path": file_path})
        
    except Exception as e:
        logger.error(f"[Memory] 写入 Debug Prompt 失败: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


# ============================================
# 接口：存入新记忆（实时对话片段），暂时关掉
# ============================================

@router.post("/memory/ingest")
async def ingest_memory(req: MemoryIngestRequest):
    """
    将一段新的对话实时存入记忆库（用于实时积累记忆）。
    
    ST 插件在 AI 回复后调用这个接口，持续把新对话喂给 FAISS。
    注意：短期内会实时追加向量，长期可以做批量压缩。
    """
    from ...rag.st_memory import ChatChunk
    
    logger.info(f"[Memory] 接收新对话片段用于存入 {req.character_name} 的记忆库, 共 {len(req.messages)} 条消息")
    
    try:
        if len(req.messages) < 2:
            return JSONResponse({"status": "skipped", "reason": "消息数量太少，不值得存入"})
        
        # 将消息列表格式化为文本块
        text_lines = []
        for msg in req.messages:
            name = msg.get("name", msg.get("role", "Unknown"))
            content = msg.get("content", "")
            text_lines.append(f"{name}: {content}")
        
        chunk_text = "\n\n".join(text_lines)
        
        chunk = ChatChunk(
            text=chunk_text,
            character_name=req.character_name,
            chat_file=req.chat_file or "realtime",
            world_line=req.world_line
        )
        
        # 在后台线程里执行（避免阻塞请求）
        import asyncio
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            memory_manager.ingest_chunks,
            [chunk],
            req.character_name,
            req.world_line
        )
        
        return JSONResponse({"status": "ok", "message": f"已将 {len(req.messages)} 条消息存入记忆库"})
        
    except Exception as e:
        logger.error(f"[Memory] 存入记忆失败: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
