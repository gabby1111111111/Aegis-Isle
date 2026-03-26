from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel

class JSONRPCRequest(BaseModel):
    jsonrpc: str = "2.0"
    id: Union[str, int]
    method: str
    params: Optional[Dict[str, Any]] = None

class JSONRPCError(BaseModel):
    code: int
    message: str
    data: Optional[Any] = None

class JSONRPCResponse(BaseModel):
    jsonrpc: str = "2.0"
    id: Union[str, int]
    result: Optional[Any] = None
    error: Optional[JSONRPCError] = None

class ToolInputSchema(BaseModel):
    type: str = "object"
    properties: Dict[str, Any]
    required: Optional[List[str]] = None

class Tool(BaseModel):
    name: str
    description: str
    inputSchema: ToolInputSchema

class TextContent(BaseModel):
    type: str = "text"
    text: str

class CallToolRequestParams(BaseModel):
    name: str
    arguments: Dict[str, Any]

class CallToolResult(BaseModel):
    content: List[TextContent]
    isError: bool = False
