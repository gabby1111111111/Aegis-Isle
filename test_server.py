"""
Aegis-Isle 测试服务器 (v2.0 - 含可观测性)

功能:
- OpenAI 兼容的 /v1/chat/completions 端点
- Shujuku 状态管理 (背包/任务/全局)
- 🆕 Token 使用统计 (tiktoken 实时计数)
- 🆕 P50/P95/P99 延迟监控
- 🆕 LLM 调用审计日志 (ELK 兼容)
- 🆕 /api/v1/metrics/* Dashboard API
"""

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
import time
import uuid

# 导入状态管理模块
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from aegis_isle.core.state.manager import StateManager
from aegis_isle.core.state.context_injection import inject_state_context, get_user_id_from_request
from aegis_isle.core.state.background_updater import update_user_state
from aegis_isle.core.state.snapshot import SnapshotManager
from aegis_isle.core.logging import logger, audit_logger
from aegis_isle.core.metrics import (
    token_metrics, TokenRecord,
    estimate_tokens, estimate_messages_tokens,
)


# 创建应用
app = FastAPI(
    title="Aegis-Isle State Management Test Server",
    description="测试服务器 v2.0 - 含状态管理 + 可观测性",
    version="0.2.0"
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
        methods = getattr(route, 'methods', None)
        if methods:
            logger.info(f"📍 Route: {route.path} {methods}")



# ============================================
# LLM 调用 (流式 + tiktoken 计数)
# ============================================

async def call_llm_streaming(messages: list, model: str = "Qwen/Qwen2.5-7B-Instruct"):
    """
    调用真实的 LLM API(流式)，同时用 tiktoken 累计 token 数。
    
    Yields:
        str: 每个文本 chunk
    
    Note:
        调用者通过 generator.prompt_tokens / generator.completion_tokens
        获取本次调用的 token 统计（在遍历完成后可用）
    """
    from openai import AsyncOpenAI
    
    logger.info(f"[LLM] 调用真实 API: model={model}, messages={len(messages)}")
    logger.debug(f"[LLM] Payload Messages: {json.dumps(messages, ensure_ascii=False)}")
    
    # 强制覆盖模型名称
    target_model = "Qwen/Qwen2.5-7B-Instruct"
    
    # 清理 messages (移除 name 字段，防止 400 错误)
    sanitized_messages = []
    for msg in messages:
        new_msg = {"role": msg["role"], "content": msg["content"]}
        sanitized_messages.append(new_msg)
    
    # 🆕 用 tiktoken 估算 prompt tokens
    prompt_tokens = estimate_messages_tokens(sanitized_messages, target_model)
    
    import os
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1")

    if not api_key:
        logger.error("[LLM] OPENAI_API_KEY 未设置，请在 .env 中配置")
        yield "[错误: 未配置 API Key]"
        return

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    completion_text = ""
    
    try:
        # 调用流式 API
        stream = await client.chat.completions.create(
            model=target_model,
            messages=sanitized_messages,
            stream=True,
            max_tokens=2000,
            temperature=0.7
        )
        
        # 流式返回，同时累积完整文本用于 token 计数
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                text = chunk.choices[0].delta.content
                completion_text += text
                yield text
                
    except Exception as e:
        logger.error(f"[LLM] API 调用失败: {e}")
        yield f"\n\n[错误: API 调用失败 - {str(e)}]"
    
    # 🆕 流式结束后，用 tiktoken 估算 completion tokens
    # 通过 generator 属性暴露给调用者
    call_llm_streaming._last_prompt_tokens = prompt_tokens
    call_llm_streaming._last_completion_tokens = estimate_tokens(completion_text, target_model)


# ============================================
# OpenAI 兼容 API
# ============================================

@app.post("/v1/chat/completions")
async def chat_completions(request: Request, background_tasks: BackgroundTasks):
    """OpenAI 兼容的聊天接口 (含 Token 统计 + 审计日志)"""
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    
    logger.info(f"🎯 [{request_id}] Endpoint hit: /v1/chat/completions")
    
    try:
        body = await request.json()

        messages = body.get("messages", [])
        stream = body.get("stream", False)
        model = body.get("model", "gpt-4")
        
        # 获取用户 ID 和角色卡信息
        user_id = get_user_id_from_request(body)
        character_card_id = body.get("character", {}).get("name", None) if isinstance(body.get("character"), dict) else None
        logger.info(f"[API] [{request_id}] 用户 {user_id} 发起对话请求")
        
        # 注入状态上下文
        state_manager = StateManager()
        user_state = await state_manager.load_state(user_id)
        state_context = state_manager.get_context_string(user_state)
        
        messages_with_state = inject_state_context(messages, state_context)
        
        # 实际使用的模型
        target_model = "Qwen/Qwen2.5-7B-Instruct"
        
        if stream:
            # 流式响应
            async def generate():
                full_response = ""
                
                async for chunk in call_llm_streaming(messages_with_state, model):
                    full_response += chunk
                    
                    # SSE 格式
                    data = {
                        "id": f"chatcmpl-{request_id}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": chunk},
                            "finish_reason": None
                        }]
                    }
                    
                    yield f"data: {json.dumps(data)}\n\n"
                
                # 🆕 流式结束：记录 Token 统计和审计日志
                latency_ms = (time.time() - start_time) * 1000
                prompt_tokens = getattr(call_llm_streaming, '_last_prompt_tokens', 0)
                completion_tokens = getattr(call_llm_streaming, '_last_completion_tokens', 0)
                
                _record_observability(
                    request_id=request_id,
                    user_id=user_id,
                    model=target_model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    latency_ms=latency_ms,
                    character_card_id=character_card_id,
                )
                
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
            
            latency_ms = (time.time() - start_time) * 1000
            prompt_tokens = getattr(call_llm_streaming, '_last_prompt_tokens', 0)
            completion_tokens = getattr(call_llm_streaming, '_last_completion_tokens', 0)
            
            # 🆕 记录 Token 统计和审计日志
            _record_observability(
                request_id=request_id,
                user_id=user_id,
                model=target_model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
                character_card_id=character_card_id,
            )
            
            # 后台更新状态
            background_tasks.add_task(
                update_user_state,
                user_id,
                full_response,
                messages[-1].get("content", "") if messages else None
            )
            
            return JSONResponse({
                "id": f"chatcmpl-{request_id}",
                "object": "chat.completion",
                "created": int(time.time()),
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
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                }
            })
            
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        logger.error(f"[API] [{request_id}] 请求处理失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 🆕 记录失败的审计日志
        audit_logger.log_llm_call(
            model="unknown",
            prompt_tokens=0,
            completion_tokens=0,
            latency_ms=latency_ms,
            user_id="unknown",
            request_id=request_id,
            outcome="error",
            error_message=str(e),
        )
        
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(e), "type": "internal_error"}}
        )


def _record_observability(
    request_id: str,
    user_id: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    latency_ms: float,
    character_card_id: str = None,
) -> None:
    """
    统一记录 Token 统计 + 审计日志。
    
    在每次 LLM 调用完成后调用，同时更新:
    1. TokenMetrics 统计面板
    2. AuditLogger 审计日志 (ELK 兼容)
    """
    # 1. Token 统计
    record = TokenRecord(
        request_id=request_id,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        latency_ms=latency_ms,
        user_id=user_id,
    )
    token_metrics.record(record)
    
    # 2. 审计日志
    audit_logger.log_llm_call(
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=latency_ms,
        user_id=user_id,
        request_id=request_id,
        character_card_id=character_card_id,
        cost_usd=record.cost_usd,
    )


# ============================================
# 🆕 Metrics Dashboard API
# ============================================

@app.get("/api/v1/metrics/dashboard")
async def metrics_dashboard():
    """
    实时 Token 统计面板。
    
    返回 Token 消耗、费用、延迟分位数和模型分布。
    """
    return JSONResponse(token_metrics.get_dashboard())


@app.get("/api/v1/metrics/token-usage")
async def metrics_token_usage(limit: int = 20):
    """最近的 Token 使用记录列表。"""
    return JSONResponse(token_metrics.get_recent_records(limit=limit))


@app.get("/api/v1/metrics/export")
async def metrics_export():
    """导出所有 Token 统计为 CSV 文件。"""
    csv_content = token_metrics.export_csv()
    return PlainTextResponse(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=token_usage.csv"}
    )


@app.post("/api/v1/metrics/reset")
async def metrics_reset():
    """重置所有统计数据。"""
    token_metrics.reset()
    return JSONResponse({"message": "Metrics reset successfully"})


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
        "version": "0.2.0",
        "features": ["stateful-agent", "token-metrics", "audit-logging", "latency-monitoring"],
        "endpoints": {
            "chat": "/v1/chat/completions",
            "state": "/v1/state/{user_id}",
            "snapshots": "/v1/state/{user_id}/snapshots",
            "rollback": "/v1/state/{user_id}/rollback",
            "metrics_dashboard": "/api/v1/metrics/dashboard",
            "metrics_token_usage": "/api/v1/metrics/token-usage",
            "metrics_export": "/api/v1/metrics/export",
        }
    }


@app.get("/health")
async def health():
    """健康检查"""
    dashboard = token_metrics.get_dashboard()
    return {
        "status": "ok",
        "service": "aegis-isle-v0.2.0",
        "total_llm_calls": dashboard["requests"]["total"],
        "total_tokens_used": dashboard["token_usage"]["total_tokens"],
    }


if __name__ == "__main__":
    import uvicorn
    print("🚀 启动 Aegis-Isle 服务器 v0.2.0 (含可观测性)...")
    print("📍 地址: http://localhost:8001")
    print("📖 API 文档: http://localhost:8001/docs")
    print("📊 Metrics: http://localhost:8001/api/v1/metrics/dashboard")
    print("")
    uvicorn.run(app, host="0.0.0.0", port=8001)
