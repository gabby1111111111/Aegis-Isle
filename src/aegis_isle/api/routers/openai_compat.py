"""
OpenAI-compatible API endpoints for Chat Completions.
Supports both streaming (SSE) and non-streaming responses.
"""

import json
import time
import uuid
from typing import List, Optional, Literal, Dict, Any, Union, AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

from ..dependencies import get_rag_pipeline
from ...rag.pipeline import RAGPipeline
from ...core.logging import logger

router = APIRouter()

# === Data Models ===

class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    name: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: str = "rag-default"
    messages: List[ChatMessage]
    stream: bool = False
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    top_p: Optional[float] = 1.0
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None

# === Helper Functions ===

def _create_chunk(
    content: Optional[str], 
    model: str, 
    finish_reason: Optional[str] = None,
    completion_id: str = None
) -> str:
    """Create a standard OpenAI SSE data chunk."""
    chunk_data = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"content": content} if content else {},
                "finish_reason": finish_reason
            }
        ]
    }
    return f"data: {json.dumps(chunk_data)}\n\n"

def _format_metadata_as_markdown(metadata_json: Dict[str, Any]) -> str:
    """Convert metadata JSON to a Markdown string prefix."""
    try:
        sources = metadata_json.get("sources", [])
        if not sources:
            return ""
        
        md_lines = ["\n> **参考文档:**"]
        for src in sources:
            doc_name = src.get('source', 'unknown')
            score = src.get('score', 0)
            md_lines.append(f"> - `{doc_name}` (相关度: {score})")
        
        md_lines.append("\n---\n")
        return "\n".join(md_lines)
    except Exception:
        return ""

async def _stream_generator(
    pipeline: RAGPipeline, 
    query: str, 
    model_name: str,
    completion_id: str,
    **kwargs
) -> AsyncGenerator[str, None]:
    """Generator for streaming responses."""
    
    # 1. Send role header (optional, mostly for structure)
    # Some clients expect the first chunk to contain role
    first_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
    }
    yield f"data: {json.dumps(first_chunk)}\n\n"

    try:
        async for chunk in pipeline.query_stream(query, **kwargs):
            content_to_send = chunk

            # Check for Metadata Packet
            # Note: pipeline yields a JSON string for metadata
            if chunk.strip().startswith('{') and '"type": "metadata"' in chunk:
                try:
                    meta_data = json.loads(chunk)
                    # Convert metadata to markdown prefix
                    content_to_send = _format_metadata_as_markdown(meta_data)
                except:
                    # If parsing fails, treat as raw text or ignore
                    pass
            
            if content_to_send:
                yield _create_chunk(content_to_send, model_name, completion_id=completion_id)

        # End of stream
        yield _create_chunk(None, model_name, finish_reason="stop", completion_id=completion_id)
        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error(f"Error in stream generator: {e}")
        error_msg = f"\n\n[System Error: {str(e)}]"
        yield _create_chunk(error_msg, model_name, finish_reason="stop", completion_id=completion_id)
        yield "data: [DONE]\n\n"

# === Route Handlers ===

@router.post("/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    """
    OpenAI-compatible Chat Completion Endpoint.
    """
    if not request.messages:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Messages list cannot be empty"
        )

    # 1. Extract Query (Last user message)
    # Simple strategy: Use the last message content as query
    # Advanced strategy could concat history, but RAG usually focuses on the latest query
    last_message = request.messages[-1]
    query = last_message.content
    
    logger.info(f"OpenAI Compat Request: {query[:50]}... (Stream={request.stream})")
    
    completion_id = f"chatcmpl-{uuid.uuid4()}"
    
    # Common kwargs for pipeline
    run_kwargs = {
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
        "top_p": request.top_p
    }

    # 2. Streaming Response
    if request.stream:
        return StreamingResponse(
            _stream_generator(
                pipeline, 
                query, 
                request.model, 
                completion_id, 
                **run_kwargs
            ),
            media_type="text/event-stream"
        )

    # 3. Non-Streaming Response
    else:
        try:
            # Call standard query method
            result = await pipeline.query(query, **run_kwargs)
            
            # Format Metadata
            metadata_prefix = ""
            if result.retrieval_result and result.retrieval_result.results:
                # Manually construct metadata packet to reuse formatter
                # Only if using standard query result structure
                # This part depends on your pipeline.query return structure
                # For now, we append sources to the end or start if desired
                # But typically non-streaming just returns clean answer
                pass 
                
            # Construct standard response
            return {
                "id": completion_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": result.answer
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": 0, # Placeholder
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error in non-streaming request: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e)
            )
