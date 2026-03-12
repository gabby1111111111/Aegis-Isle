"""
OpenAI 兼容接口 (集成状态管理 + 真实 SiliconFlow LLM)

修改点:
1. 引入 BackgroundTasks
2. 集成 context_injection
3. 集成 background_updater
4. 🆕 接入真实 SiliconFlow API (替换 Mock)
"""

from fastapi import APIRouter, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from typing import AsyncGenerator
import json
import asyncio
from openai import AsyncOpenAI

from ...core.logging import logger
from ...core.config import settings
from ...core.state.manager import StateManager
from ...core.state.context_injection import (
    inject_state_context,
    get_user_id_from_request,
)
from ...core.state.background_updater import update_user_state
from ...core.state.snapshot import SnapshotManager
from ...rag.st_memory_manager import memory_manager
import hashlib
import os
import re


# Mock references to LangGraph for Task C trigger (to be implemented more fully if needed)
# For now, we'll represent the triggering of a background agent chain.
async def trigger_agent_chain(
    command: str, user_id: str, universe_id: str, character_name: str, messages: list
):
    logger.info(
        f"[AgentTrigger] Background task started for command {command} "
        f"in universe {universe_id} for char {character_name}"
    )
    # TODO: Initialize GraphState, route to specific agents, and save results to memory.


def get_universe_id(messages: list, character_name: str = "Unknown") -> str:
    """Extract universe_id from ST System Prompt or generate a fallback."""
    for msg in messages:
        if msg.get("role") == "system":
            content = msg.get("content", "")
            m = re.search(r"\[Universe:\s*(.+?)\]", content)
            if m:
                return m.group(1).strip()

            # Fallback: Hash of character_name + first 100 chars of system prompt
            hash_input = f"{character_name}_{content[:100]}".encode("utf-8")
            fallback_id = (
                f"fallback_{character_name}_{hashlib.md5(hash_input).hexdigest()[:8]}"
            )
            return fallback_id
    return f"fallback_{character_name}_default"


router = APIRouter()

# 企业版强制使用的模型 (可通过 .env 的 DEFAULT_LLM_MODEL 覆盖)
TARGET_MODEL = "Qwen/Qwen2.5-7B-Instruct"


# ============================================
# 真实 LLM 调用 (SiliconFlow API)
# ============================================


async def call_llm_streaming(
    messages: list, model: str = "gpt-4"
) -> AsyncGenerator[str, None]:
    """
    调用真实的 SiliconFlow LLM API (流式)

    从 settings 读取 API Key 和 Base URL:
      - OPENAI_API_KEY  → SiliconFlow API Key
      - OPENAI_BASE_URL → https://api.siliconflow.cn/v1

    强制覆盖模型为 TARGET_MODEL，兼容 SillyTavern 传来的任意模型名。
    """

    logger.info(f"[LLM] 调用 SiliconFlow API: model={model}, messages={len(messages)}")

    # 清理 messages，移除 SillyTavern 可能附带的 name 字段（会导致 400）
    sanitized_messages = [
        {"role": msg["role"], "content": msg.get("content", "")} for msg in messages
    ]

    # 从 settings 读取（来自 .env 的 OPENAI_API_KEY / OPENAI_BASE_URL）
    api_key = settings.openai_api_key

    if not api_key:
        logger.error("[LLM] OPENAI_API_KEY 未配置，请在 .env 文件中设置")
        yield "[错误: 未配置 API Key，请在 .env 中设置 OPENAI_API_KEY]"
        return

    try:
        client = AsyncOpenAI(
            api_key=api_key, base_url=settings.openai_base_url
        )
        stream = await client.chat.completions.create(
            model=TARGET_MODEL,
            messages=sanitized_messages,
            stream=True,
            max_tokens=2000,
            temperature=settings.temperature,
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    except Exception as e:
        logger.error(f"[LLM] SiliconFlow API 调用失败: {e}")
        yield f"\n\n[错误: LLM API 调用失败 - {str(e)}]"


# ============================================
# API 端点
# ============================================


@router.post("/chat/completions")
async def chat_completions(request: Request, background_tasks: BackgroundTasks):
    """
    OpenAI 兼容的聊天完成接口(集成状态管理)

    工作流程:
    1. 提取用户 ID
    2. 加载用户状态
    3. 注入状态到 Prompt
    4. 调用 LLM
    5. 流式返回响应
    6. **后台异步更新状态**
    """
    try:
        # 解析请求
        body = await request.json()
        messages = body.get("messages", [])
        
        if not messages:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": "messages array is required and cannot be empty", "type": "invalid_request_error"}},
            )
            
        model = body.get("model", "gpt-4")
        stream = body.get("stream", True)

        # 提取用户 ID
        user_id = get_user_id_from_request(body)
        logger.info(f"[API] 处理用户 {user_id} 的请求")

        # 加载用户状态
        state_manager = StateManager()
        user_state = await state_manager.load_state(user_id)
        logger.info(f"[API] 已加载用户 {user_id} 的状态")

        # 获取状态的 Markdown 表示
        state_context = state_manager.get_context_string(user_state)

        # 注入状态到消息列表
        enhanced_messages = inject_state_context(
            messages=messages, state_markdown=state_context, max_tokens=2000
        )

        # 获取用户最新消息(用于日志和检索记忆)
        user_message = messages[-1].get("content", "") if messages else ""

        # 🌟 任务 C: ST 触发指令拦截
        target_character = body.get(
            "character", "Unknown"
        )  # Extract character name from ST request if possible
        world_line = get_universe_id(messages, target_character)

        if user_message.strip().startswith("/"):
            command = user_message.strip().split(" ")[0].lower()
            if command in ["/recap", "/relation", "/portrait", "/plot"]:
                # 立即返回占位符, 阻断主链路 LLM 调用
                logger.info(f"[API] 拦截到 ST 指令 {command}，启动后台 Agent 链...")
                background_tasks.add_task(
                    trigger_agent_chain,
                    command=command,
                    user_id=user_id,
                    universe_id=world_line,
                    character_name=target_character,
                    messages=enhanced_messages,
                )

                placeholder_text = "⏳ 正在分析，结果将在下一轮对话中呈现..."
                if stream:

                    async def command_placeholder_stream():
                        data = {
                            "id": f"chatcmpl-{user_id}",
                            "object": "chat.completion.chunk",
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": placeholder_text},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                        yield "data: [DONE]\n\n"

                    return StreamingResponse(
                        command_placeholder_stream(), media_type="text/event-stream"
                    )
                else:
                    return JSONResponse(
                        {
                            "id": f"chatcmpl-{user_id}",
                            "object": "chat.completion",
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": placeholder_text,
                                    },
                                    "finish_reason": "stop",
                                }
                            ],
                        }
                    )

        # 🌟 任务 A: 检索路由 & 并发获取记忆
        # 并发任务定义
        async def fetch_faiss_chunks(query, char, world):
            # Fallback mock for faiss sub_chunks logic
            return await memory_manager.search_memory(query, char, world, k=3)

        async def fetch_graph_memory(query, char, world):
            # TODO: Add real graph edge/node searching here based on keywords
            # Mock empty for now
            return []

        async def fetch_episode_memory(query, char, world):
            # TODO: Add real episodes searching here
            return []

        async def _empty_list():
            return []

        if user_message:
            try:
                # 1. 意图检测 (纯关键词)
                do_faiss = any(
                    k in user_message for k in ["那段", "那时候", "当时", "氛围"]
                )
                do_graph = any(
                    k in user_message for k in ["关系", "感觉", "对我", "喜不喜欢"]
                )
                do_episode = any(
                    k in user_message for k in ["第一次", "什么时候", "发生过"]
                )

                # Default to FAISS + Episode if no specific keyword matched
                if not do_faiss and not do_graph and not do_episode:
                    do_faiss = True
                    do_episode = True

                tasks = []
                # 2. 路由分发并发
                if do_faiss:
                    tasks.append(
                        fetch_faiss_chunks(user_message, target_character, world_line)
                    )
                else:
                    tasks.append(_empty_list())

                if do_graph:
                    tasks.append(
                        fetch_graph_memory(user_message, target_character, world_line)
                    )
                else:
                    tasks.append(_empty_list())

                if do_episode:
                    tasks.append(
                        fetch_episode_memory(user_message, target_character, world_line)
                    )
                else:
                    tasks.append(_empty_list())

                results = await asyncio.gather(*tasks)
                faiss_docs, graph_docs, episode_docs = results

                # Combine results
                all_docs = faiss_docs + graph_docs + episode_docs

                if all_docs:
                    memory_context = memory_manager.format_context_for_prompt(all_docs)
                    logger.info(
                        f"[API] [{world_line}] 检索出 {len(all_docs)} 块长线记忆，准备注入。并发耗时极低。"
                    )

                    # 3. 将长期记忆拼接在 System Message 的最后
                    for msg in enhanced_messages:
                        if msg["role"] == "system":
                            msg["content"] += f"\n\n{memory_context}"
                            break
            except Exception as e:
                logger.error(f"[API] 注入长线记忆时发生错误: {e}")

        # 🌟 调试输出保存
        if os.environ.get("DEBUG_SAVE", "").lower() == "true":
            from datetime import datetime
            from pathlib import Path

            debug_dir = Path("debug/prompts")
            debug_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_file = debug_dir / f"{world_line}_{ts}.txt"
            with open(debug_file, "w", encoding="utf-8") as f:
                f.write("=== ENHANCED MESSAGES ===\n")
                f.write(json.dumps(enhanced_messages, ensure_ascii=False, indent=2))

        logger.info(f"[API] 状态上下文长度: {len(state_context)} 字符")
        logger.debug(f"[API] 增强后消息数: {len(enhanced_messages)}")

        # 此时已经获取了用户消息

        # 流式响应
        if stream:

            async def generate_sse_stream():
                """生成 SSE 格式的流式响应"""
                full_response = ""

                try:
                    # 调用 LLM
                    async for chunk in call_llm_streaming(enhanced_messages, model):
                        full_response += chunk

                        # 构造 SSE 数据
                        data = {
                            "id": f"chatcmpl-{user_id}",
                            "object": "chat.completion.chunk",
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": chunk},
                                    "finish_reason": None,
                                }
                            ],
                        }

                        yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

                    # 发送结束标记
                    yield "data: [DONE]\n\n"

                    logger.info(f"[API] 流式响应完成,总长度: {len(full_response)} 字符")

                    # 🔥 触发后台状态更新
                    background_tasks.add_task(
                        update_user_state,
                        user_id=user_id,
                        llm_output=full_response,
                        user_message=user_message,
                    )

                except Exception as e:
                    logger.error(f"[API] 流式响应异常: {e}")
                    error_data = {"error": str(e)}
                    yield f"data: {json.dumps(error_data)}\n\n"

            return StreamingResponse(
                generate_sse_stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        # 非流式响应
        else:
            full_response = ""
            async for chunk in call_llm_streaming(enhanced_messages, model):
                full_response += chunk

            logger.info(f"[API] 非流式响应完成,长度: {len(full_response)} 字符")

            # 🔥 触发后台状态更新
            background_tasks.add_task(
                update_user_state,
                user_id=user_id,
                llm_output=full_response,
                user_message=user_message,
            )

            return JSONResponse(
                {
                    "id": f"chatcmpl-{user_id}",
                    "object": "chat.completion",
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": full_response},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": len(json.dumps(enhanced_messages)),
                        "completion_tokens": len(full_response),
                        "total_tokens": len(json.dumps(enhanced_messages))
                        + len(full_response),
                    },
                }
            )

    except Exception as e:
        logger.error(f"[API] 请求处理失败: {e}")
        import traceback

        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(e), "type": "internal_error"}},
        )


@router.get("/state/{user_id}")
async def get_user_state_debug(user_id: str):
    """
    调试接口:查看用户状态

    Args:
        user_id: 用户 ID

    Returns:
        用户状态的 JSON 表示
    """
    try:
        state_manager = StateManager()
        user_state = await state_manager.load_state(user_id)

        return JSONResponse(
            {
                "user_id": user_state.user_id,
                "version": user_state.version,
                "sheets_summary": {
                    uid: {
                        "name": sheet.name,
                        "row_count": len(sheet.get_rows()),
                        "order": sheet.order_no,
                    }
                    for uid, sheet in user_state.sheets.items()
                },
            }
        )
    except Exception as e:
        logger.error(f"[API] 获取状态失败: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.get("/state/{user_id}/snapshots")
async def list_user_snapshots(user_id: str, limit: int = 10):
    """
    列出用户的所有快照

    Args:
        user_id: 用户 ID
        limit: 返回数量限制

    Returns:
        快照列表(JSON 格式)

    Example:
        GET /v1/state/test_user/snapshots?limit=5
    """
    try:
        snapshot_manager = SnapshotManager()
        snapshots = await snapshot_manager.list_snapshots(user_id, limit=limit)

        return JSONResponse(
            {
                "success": True,
                "user_id": user_id,
                "snapshot_count": len(snapshots),
                "snapshots": [
                    {
                        "snapshot_id": snap.snapshot_id,
                        "timestamp": snap.timestamp.isoformat(),
                        "version": snap.version,
                        "change_summary": snap.change_summary,
                        "file_path": snap.file_path,
                    }
                    for snap in snapshots
                ],
            }
        )
    except Exception as e:
        logger.error(f"[API] 获取快照列表失败: {e}")
        return JSONResponse(
            status_code=500, content={"success": False, "error": str(e)}
        )


@router.post("/state/{user_id}/rollback")
async def rollback_user_state(user_id: str, request: Request):
    """
    回滚用户状态到指定快照

    Args:
        user_id: 用户 ID
        request: 包含 snapshot_id 的 JSON 请求

    Request Body:
        {
            "snapshot_id": "snap_20260206_120000"
        }

    Returns:
        回滚结果

    Example:
        POST /v1/state/test_user/rollback
        Body: {"snapshot_id": "snap_20260206_120000"}
    """
    try:
        body = await request.json()
        snapshot_id = body.get("snapshot_id")

        if not snapshot_id:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "snapshot_id is required"},
            )

        snapshot_manager = SnapshotManager()

        # 回滚到快照
        restored_state = await snapshot_manager.rollback_to_snapshot(
            user_id, snapshot_id
        )

        if not restored_state:
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "error": f"Snapshot {snapshot_id} not found or rollback failed",
                },
            )

        # 保存回滚后的状态
        state_manager = StateManager()
        success = await state_manager.save_state(restored_state)

        if success:
            logger.info(f"[API] 用户 {user_id} 已回滚到快照 {snapshot_id}")

            return JSONResponse(
                {
                    "success": True,
                    "user_id": user_id,
                    "snapshot_id": snapshot_id,
                    "restored_version": restored_state.version,
                    "message": f"Successfully rolled back to snapshot {snapshot_id}",
                }
            )
        else:
            return JSONResponse(
                status_code=500,
                content={"success": False, "error": "Failed to save rolled back state"},
            )

    except Exception as e:
        logger.error(f"[API] 回滚失败: {e}")
        import traceback

        traceback.print_exc()

        return JSONResponse(
            status_code=500, content={"success": False, "error": str(e)}
        )


@router.delete("/state/{user_id}")
async def delete_user_state(user_id: str):
    """
    删除用户状态(调试用)

    Args:
        user_id: 用户 ID

    Returns:
        删除结果
    """
    try:
        state_manager = StateManager()
        success = state_manager.delete(user_id, create_backup=True)

        if success:
            return JSONResponse(
                {
                    "success": True,
                    "message": f"User {user_id} state deleted (backup created)",
                }
            )
        else:
            return JSONResponse(
                status_code=500,
                content={"success": False, "error": "Failed to delete state"},
            )
    except Exception as e:
        logger.error(f"[API] 删除状态失败: {e}")
        return JSONResponse(
            status_code=500, content={"success": False, "error": str(e)}
        )


# 导出
__all__ = ["router"]
