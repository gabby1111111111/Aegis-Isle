from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import json
import logging

from aegis_isle.mcp.protocol import JSONRPCRequest
from aegis_isle.mcp.handler import mcp_handler

logger = logging.getLogger(__name__)

mcp_app = FastAPI(title="Aegis-Isle MCP Server", description="Model Context Protocol endpoints for Aegis-Isle")

@mcp_app.websocket("/ws")
async def mcp_websocket(websocket: WebSocket):
    await websocket.accept()
    logger.info("MCP WebSocket client connected.")
    try:
        while True:
            data = await websocket.receive_text()
            try:
                req_dict = json.loads(data)
                request = JSONRPCRequest(**req_dict)
            except Exception as e:
                logger.error(f"Failed to parse MCP request: {e}")
                err_response = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}
                await websocket.send_text(json.dumps(err_response))
                continue
                
            response = await mcp_handler.handle_request(request)
            await websocket.send_text(response.model_dump_json(exclude_none=True))
            
    except WebSocketDisconnect:
        logger.info("MCP WebSocket client disconnected.")
    except Exception as e:
        logger.error(f"Unexpected error in MCP WebSocket: {e}", exc_info=True)

@mcp_app.post("/http")
async def mcp_http(request: JSONRPCRequest):
    response = await mcp_handler.handle_request(request)
    return JSONResponse(content=response.model_dump(exclude_none=True))
