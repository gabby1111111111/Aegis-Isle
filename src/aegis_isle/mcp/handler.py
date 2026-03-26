import httpx
import logging
from typing import Dict, Any, Optional, Union

from aegis_isle.core.config import settings
from aegis_isle.rag.st_memory_manager import memory_manager
from aegis_isle.mcp.protocol import (
    Tool, ToolInputSchema, CallToolResult, TextContent,
    JSONRPCRequest, JSONRPCResponse, JSONRPCError
)

logger = logging.getLogger(__name__)

# Register Tools
AVAILABLE_TOOLS = [
    Tool(
        name="send_ntfy_notification",
        description="向 Gabby 大人的手机发送物理振铃的紧急通知，用于必须打断她或者提醒她看 ST 屏幕的场合。",
        inputSchema=ToolInputSchema(
            properties={
                "message": {
                    "type": "string",
                    "description": "要发送的通知文本内容"
                }
            },
            required=["message"]
        )
    ),
    Tool(
        name="search_aegis_memory",
        description="基于多宇宙的 RAG 记忆检索，从 Aegis-Isle 的历史对话 FAISS 索引中提取之前的记忆上下文。",
        inputSchema=ToolInputSchema(
            properties={
                "query": {
                    "type": "string",
                    "description": "搜索的问题或关键词"
                },
                "character_name": {
                    "type": "string",
                    "description": "对话的角色名（如 'Gabriella', 'Astarion'）"
                },
                "world_line": {
                    "type": "string",
                    "description": "可选的宇宙/世界线标识，如果有多个可以用逗号分隔。不填则使用基准宇宙。"
                }
            },
            required=["query", "character_name"]
        )
    )
]

class MCPServerHandler:
    def __init__(self):
        self.tools = {tool.name: tool for tool in AVAILABLE_TOOLS}

    async def handle_request(self, request: JSONRPCRequest) -> JSONRPCResponse:
        try:
            if request.method == "tools/list":
                return JSONRPCResponse(
                    id=request.id,
                    result={"tools": [t.model_dump() for t in AVAILABLE_TOOLS]}
                )
            
            elif request.method == "tools/call":
                if not request.params or "name" not in request.params:
                    return self._error(request.id, -32602, "Invalid params for tools/call")
                
                tool_name = request.params["name"]
                arguments = request.params.get("arguments", {})
                
                result = await self._execute_tool(tool_name, arguments)
                return JSONRPCResponse(id=request.id, result=result.model_dump())
            
            else:
                return self._error(request.id, -32601, f"Method not found: {request.method}")
                
        except Exception as e:
            logger.error(f"MCP Handler Error: {e}", exc_info=True)
            return self._error(request.id, -32000, str(e))

    async def _execute_tool(self, name: str, arguments: Dict[str, Any]) -> CallToolResult:
        if name == "send_ntfy_notification":
            message = arguments.get("message", "Empty Message")
            return await self._tool_send_ntfy(message)
        
        elif name == "search_aegis_memory":
            query = arguments.get("query")
            character_name = arguments.get("character_name")
            world_line = arguments.get("world_line")
            if not query or not character_name:
                return CallToolResult(
                    content=[TextContent(text="Error: Missing query or character_name.")],
                    isError=True
                )
            return await self._tool_search_memory(query, character_name, world_line)
            
        else:
            return CallToolResult(
                content=[TextContent(text=f"Unknown tool: {name}")],
                isError=True
            )

    async def _tool_send_ntfy(self, message: str) -> CallToolResult:
        ntfy_url = f"https://ntfy.sh/{settings.ntfy_topic_ring}"
        headers = {"Title": "Aegis-Isle MCP Notification", "Priority": "high"}
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(ntfy_url, content=message.encode("utf-8"), headers=headers)
                resp.raise_for_status()
            return CallToolResult(
                content=[TextContent(text=f"Successfully sent ntfy notification: {message}")],
                isError=False
            )
        except Exception as e:
            logger.error(f"Failed to send ntfy notification: {e}")
            return CallToolResult(
                content=[TextContent(text=f"Failed to send ntfy notification. Error: {e}")],
                isError=True
            )

    async def _tool_search_memory(self, query: str, character_name: str, world_line: Optional[str] = None) -> CallToolResult:
        try:
            docs = await memory_manager.search_memory(
                query=query, 
                character_name=character_name, 
                world_line=world_line, 
                k=4
            )
            
            if not docs:
                return CallToolResult(
                    content=[TextContent(text="No relevant memories found.")],
                    isError=False
                )
                
            formatted_context = memory_manager.format_context_for_prompt(docs)
            return CallToolResult(
                content=[TextContent(text=formatted_context)],
                isError=False
            )
        except Exception as e:
             logger.error(f"Memory search failed: {e}")
             return CallToolResult(
                 content=[TextContent(text=f"Memory search failed. Error: {e}")],
                 isError=True
             )

    def _error(self, req_id: Union[str, int], code: int, message: str) -> JSONRPCResponse:
        return JSONRPCResponse(
            id=req_id,
            error=JSONRPCError(code=code, message=message)
        )

# Global singleton
mcp_handler = MCPServerHandler()
