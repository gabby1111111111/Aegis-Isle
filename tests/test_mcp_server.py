import pytest
from fastapi.testclient import TestClient
from src.aegis_isle.api.main import app

client = TestClient(app)

def test_mcp_http_tools_list():
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/list"
    }
    response = client.post("/mcp/http", json=request)
    assert response.status_code == 200
    data = response.json()
    assert data["jsonrpc"] == "2.0"
    assert data["id"] == 1
    assert "result" in data
    assert "tools" in data["result"]
    
    # Check if our custom tools are registered
    tool_names = [t["name"] for t in data["result"]["tools"]]
    assert "send_ntfy_notification" in tool_names
    assert "search_aegis_memory" in tool_names

def test_mcp_websocket_connection():
    with client.websocket_connect("/mcp/ws") as websocket:
        request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list"
        }
        websocket.send_json(request)
        data = websocket.receive_json()
        assert data["jsonrpc"] == "2.0"
        assert data["id"] == 2
        assert "result" in data
        assert "tools" in data["result"]
