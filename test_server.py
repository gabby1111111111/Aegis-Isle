"""
简化测试服务器 - 仅用于状态管理测试
跳过 RAG pipeline 初始化
"""

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json

# 导入状态管理模块
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from aegis_isle.core.state.manager import StateManager
from aegis_isle.core.state.context_injection import inject_state_context, get_user_id_from_request
from aegis_isle.core.state.background_updater import update_user_state
from aegis_isle.core.state.snapshot import SnapshotManager
from aegis_isle.core.logging import logger


# 创建应用
app = FastAPI(
    title="Aegis-Isle State Management Test Server",
    description="简化测试服务器 - 仅用于状态管理功能测试",
    version="0.1.0-test"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Server Startup: Listing Routes")
    for route in app.routes:
        logger.info(f"📍 Route: {route.path} {route.methods}")



# ============================================
# 模拟 LLM 调用
# ============================================

async def call_llm_streaming(messages: list, model: str = "Qwen/Qwen2.5-7B-Instruct"):
    """
    调用真实的 LLM API(流式)
    """
    from openai import AsyncOpenAI
    from aegis_isle.core.config import settings
    
    # 强制使用 SiliconFlow 配置
    # API Key: sk-enrrsvuvlvaztjmzilcxnofmowvttxsxydbosovlknmgqhar
    # Base URL: https://api.siliconflow.cn/v1
    
    logger.info(f"[LLM] 调用真实 API: model={model}, messages={len(messages)}")
    logger.debug(f"[LLM] Payload Messages: {json.dumps(messages, ensure_ascii=False)}")
    
    # 强制覆盖模型名称，因为 SillyTavern 可能会传 gpt-4
    target_model = "Qwen/Qwen2.5-7B-Instruct"
    
    # 清理 messages (移除 name 字段，防止 400 错误)
    sanitized_messages = []
    for msg in messages:
        new_msg = {"role": msg["role"], "content": msg["content"]}
        sanitized_messages.append(new_msg)
    
    client = AsyncOpenAI(
        api_key="sk-enrrsvuvlvaztjmzilcxnofmowvttxsxydbosovlknmgqhar",
        base_url="https://api.siliconflow.cn/v1"
    )
    
    try:
        # 调用流式 API
        stream = await client.chat.completions.create(
            model=target_model,
            messages=sanitized_messages,
            stream=True,
            max_tokens=2000,
            temperature=0.7
        )
        
        # 流式返回
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
                
    except Exception as e:
        logger.error(f"[LLM] API 调用失败: {e}")
        yield f"\n\n[错误: API 调用失败 - {str(e)}]"


# ============================================
# OpenAI 兼容 API
# ============================================

@app.post("/v1/chat/completions")
async def chat_completions(request: Request, background_tasks: BackgroundTasks):
    """OpenAI 兼容的聊天接口"""
    logger.info("🎯 Endpoint hit: /v1/chat/completions")
    print("🎯 Endpoint hit: /v1/chat/completions")
    try:
        body = await request.json()

        messages = body.get("messages", [])
        stream = body.get("stream", False)
        model = body.get("model", "gpt-4")
        
        # 获取用户 ID
        user_id = get_user_id_from_request(body)
        logger.info(f"[API] 用户 {user_id} 发起对话请求")
        
        # 注入状态上下文
        state_manager = StateManager()
        user_state = await state_manager.load_state(user_id)
        state_context = state_manager.get_context_string(user_state)
        
        messages_with_state = inject_state_context(messages, state_context)
        
        if stream:
            # 流式响应
            async def generate():
                full_response = ""
                
                async for chunk in call_llm_streaming(messages_with_state, model):
                    full_response += chunk
                    
                    # SSE 格式
                    data = {
                        "id": "chatcmpl-test",
                        "object": "chat.completion.chunk",
                        "created": 1234567890,
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": chunk},
                            "finish_reason": None
                        }]
                    }
                    
                    yield f"data: {json.dumps(data)}\n\n"
                
                # 后台更新状态
                background_tasks.add_task(
                    update_user_state,
                    user_id,
                    full_response,
                    messages[-1].get("content", "") if messages else None
                )
                
                # 结束标记
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(generate(), media_type="text/event-stream")
        
        else:
            # 非流式响应
            full_response = ""
            async for chunk in call_llm_streaming(messages_with_state, model):
                full_response += chunk
            
            # 后台更新状态
            background_tasks.add_task(
                update_user_state,
                user_id,
                full_response,
                messages[-1].get("content", "") if messages else None
            )
            
            return JSONResponse({
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1234567890,
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": full_response
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150
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


# ============================================
# 状态管理 API
# ============================================

@app.get("/v1/state/{user_id}")
async def get_user_state_debug(user_id: str):
    """查看用户状态"""
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


@app.get("/v1/state/{user_id}/snapshots")
async def list_user_snapshots(user_id: str, limit: int = 10):
    """列出用户快照"""
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
            content={"success": False, "error": str(e)}
        )


@app.post("/v1/state/{user_id}/rollback")
async def rollback_user_state(user_id: str, request: Request):
    """回滚用户状态"""
    try:
        body = await request.json()
        snapshot_id = body.get("snapshot_id")
        
        if not snapshot_id:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "snapshot_id is required"}
            )
        
        snapshot_manager = SnapshotManager()
        restored_state = await snapshot_manager.rollback_to_snapshot(user_id, snapshot_id)
        
        if not restored_state:
            return JSONResponse(
                status_code=404,
                content={"success": False, "error": f"Snapshot {snapshot_id} not found"}
            )
        
        state_manager = StateManager()
        success = await state_manager.save_state(restored_state)
        
        if success:
            logger.info(f"[API] 用户 {user_id} 已回滚到快照 {snapshot_id}")
            return JSONResponse({
                "success": True,
                "user_id": user_id,
                "snapshot_id": snapshot_id,
                "restored_version": restored_state.version
            })
        else:
            return JSONResponse(
                status_code=500,
                content={"success": False, "error": "Failed to save rolled back state"}
            )
        
    except Exception as e:
        logger.error(f"[API] 回滚失败: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )


@app.get("/")
async def root():
    """根端点"""
    return {
        "message": "Aegis-Isle State Management Test Server",
        "version": "0.1.0-test",
        "endpoints": {
            "chat": "/v1/chat/completions",
            "state": "/v1/state/{user_id}",
            "snapshots": "/v1/state/{user_id}/snapshots",
            "rollback": "/v1/state/{user_id}/rollback"
        }
    }


@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "ok", "service": "state-management-test"}


if __name__ == "__main__":
    import uvicorn
    print("🚀 启动 Aegis-Isle 状态管理测试服务器...")
    print("📍 地址: http://localhost:8001")
    print("📖 API 文档: http://localhost:8001/docs")
    print("")
    uvicorn.run(app, host="0.0.0.0", port=8001)
