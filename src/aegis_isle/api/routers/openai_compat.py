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
import time
import uuid

from ...core.logging import logger
from ...core.config import settings
from ...core.state.manager import StateManager
from ...core.state.context_injection import inject_state_context, get_user_id_from_request
from ...core.state.background_updater import update_user_state
from ...core.state.snapshot import SnapshotManager


router = APIRouter()

# 企业版强制使用的模型 (可通过 .env 的 DEFAULT_LLM_MODEL 覆盖)
TARGET_MODEL = "Qwen/Qwen2.5-7B-Instruct"


# ============================================
# 真实 LLM 调用 (SiliconFlow API)
# ============================================

async def call_llm_streaming(
    messages: list,
    model: str = "gpt-4"
) -> AsyncGenerator[str, None]:
    """
    调用真实的 SiliconFlow LLM API (流式)

    从 settings 读取 API Key 和 Base URL:
      - OPENAI_API_KEY  → SiliconFlow API Key
      - OPENAI_BASE_URL → https://api.siliconflow.cn/v1

    强制覆盖模型为 TARGET_MODEL，兼容 SillyTavern 传来的任意模型名。
    """
    from openai import AsyncOpenAI

    logger.info(f"[LLM] 调用 SiliconFlow API: model={model}, messages={len(messages)}")

    # 清理 messages，移除 SillyTavern 可能附带的 name 字段（会导致 400）
    sanitized_messages = [
        {"role": msg["role"], "content": msg.get("content", "")}
        for msg in messages
    ]

    # 从 settings 读取（来自 .env 的 OPENAI_API_KEY / OPENAI_BASE_URL）
    api_key = settings.openai_api_key
    base_url = settings.openai_base_url or "https://api.siliconflow.cn/v1"

    if not api_key:
        logger.error("[LLM] OPENAI_API_KEY 未配置，请在 .env 文件中设置")
        yield "[错误: 未配置 API Key，请在 .env 中设置 OPENAI_API_KEY]"
        return

    try:
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
async def chat_completions(
    request: Request,
    background_tasks: BackgroundTasks
):
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
            messages=messages,
            state_markdown=state_context,
            max_tokens=2000
        )
        
        logger.info(f"[API] 状态上下文长度: {len(state_context)} 字符")
        logger.debug(f"[API] 增强后消息数: {len(enhanced_messages)}")
        
        # 获取用户最新消息(用于日志)
        user_message = messages[-1].get("content", "") if messages else ""
        
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
                                    "finish_reason": None
                                }
                            ]
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
                        user_message=user_message
                    )
                    
                except Exception as e:
                    logger.error(f"[API] 流式响应异常: {e}")
                    error_data = {"error": str(e)}
                    yield f"data: {json.dumps(error_data)}\n\n"
            
            return StreamingResponse(
                generate_sse_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no"
                }
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
                user_message=user_message
            )
            
            return JSONResponse({
                "id": f"chatcmpl-{user_id}",
                "object": "chat.completion",
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": full_response
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": len(json.dumps(enhanced_messages)),
                    "completion_tokens": len(full_response),
                    "total_tokens": len(json.dumps(enhanced_messages)) + len(full_response)
                }
            })
            
    except Exception as e:
        logger.error(f"[API] 请求处理失败: {e}")
        import traceback
        traceback.print_exc()
        
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(e), "type": "internal_error"}}
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
        
        return JSONResponse({
            "user_id": user_state.user_id,
            "version": user_state.version,
            "sheets_summary": {
                uid: {
                    "name": sheet.name,
                    "row_count": len(sheet.get_rows()),
                    "order": sheet.order_no
                }
                for uid, sheet in user_state.sheets.items()
            }
        })
    except Exception as e:
        logger.error(f"[API] 获取状态失败: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


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
        
        return JSONResponse({
            "success": True,
            "user_id": user_id,
            "snapshot_count": len(snapshots),
            "snapshots": [
                {
                    "snapshot_id": snap.snapshot_id,
                    "timestamp": snap.timestamp.isoformat(),
                    "version": snap.version,
                    "change_summary": snap.change_summary,
                    "file_path": snap.file_path
                }
                for snap in snapshots
            ]
        })
    except Exception as e:
        logger.error(f"[API] 获取快照列表失败: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )


@router.post("/state/{user_id}/rollback")
async def rollback_user_state(
    user_id: str,
    request: Request
):
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
                content={
                    "success": False,
                    "error": "snapshot_id is required"
                }
            )
        
        snapshot_manager = SnapshotManager()
        
        # 回滚到快照
        restored_state = await snapshot_manager.rollback_to_snapshot(
            user_id, 
            snapshot_id
        )
        
        if not restored_state:
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "error": f"Snapshot {snapshot_id} not found or rollback failed"
                }
            )
        
        # 保存回滚后的状态
        state_manager = StateManager()
        success = await state_manager.save_state(restored_state)
        
        if success:
            logger.info(f"[API] 用户 {user_id} 已回滚到快照 {snapshot_id}")
            
            return JSONResponse({
                "success": True,
                "user_id": user_id,
                "snapshot_id": snapshot_id,
                "restored_version": restored_state.version,
                "message": f"Successfully rolled back to snapshot {snapshot_id}"
            })
        else:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": "Failed to save rolled back state"
                }
            )
        
    except Exception as e:
        logger.error(f"[API] 回滚失败: {e}")
        import traceback
        traceback.print_exc()
        
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
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
            return JSONResponse({
                "success": True,
                "message": f"User {user_id} state deleted (backup created)"
            })
        else:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": "Failed to delete state"
                }
            )
    except Exception as e:
        logger.error(f"[API] 删除状态失败: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )


# 导出
__all__ = ["router"]
