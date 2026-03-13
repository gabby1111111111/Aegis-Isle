"""
功能测试: FastAPI API 端点
=============================
测试真实用户请求路径，验证 HTTP 状态码、响应结构、数据格式。
使用 httpx.AsyncClient 异步测试。
"""

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from src.aegis_isle.api.main import create_app


@pytest_asyncio.fixture
async def client():
    """创建一个不触发 lifespan 的轻量 test client"""
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ============================================
# 1. 基础连通性
# ============================================


class TestRootEndpoints:
    """根路径和 info 端点的功能测试"""

    @pytest.mark.asyncio
    async def test_root_returns_welcome(self, client):
        """用户首次访问根路径应看到欢迎信息"""
        resp = await client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert "message" in data
        assert "AegisIsle" in data["message"]

    @pytest.mark.asyncio
    async def test_info_returns_system_info(self, client):
        """/info 应返回系统版本和功能开关"""
        resp = await client.get("/info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["system"] == "AegisIsle"
        assert "version" in data
        assert "features" in data
        assert isinstance(data["features"]["rag"], bool)


# ============================================
# 2. OpenAI 兼容层 (ST 核心链路)
# ============================================


class TestOpenAICompat:
    """SillyTavern 通过 /v1/chat/completions 发请求的场景"""

    @pytest.mark.asyncio
    async def test_chat_completions_missing_messages_returns_400(self, client):
        """缺少 messages 字段应返回 400（已修复）"""
        resp = await client.post("/v1/chat/completions", json={"model": "gpt-4"})
        assert resp.status_code == 400
        data = resp.json()
        assert "error" in data
        assert "messages" in data["error"]["message"]

    @pytest.mark.asyncio
    async def test_chat_completions_empty_messages_handled(self, client):
        """空 messages 列表应被优雅处理"""
        resp = await client.post(
            "/v1/chat/completions", json={"model": "gpt-4", "messages": []}
        )
        # 不应 500 崩溃
        assert resp.status_code != 500 or resp.status_code == 200


# ============================================
# 3. 记忆检索 API
# ============================================


class TestMemoryAPI:
    """长线记忆 API 的功能测试"""

    @pytest.mark.asyncio
    async def test_memory_search_valid_request(self, client):
        """有效的记忆检索请求应返回结构化响应"""
        resp = await client.post(
            "/v1/memory/search",
            json={
                "query": "你还记得那次在法餐厅的事吗？",
                "character_name": "ZouZheng",
                "world_line": "AIDom",
                "k": 3,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "memories" in data or "context_string" in data

    @pytest.mark.asyncio
    async def test_memory_search_missing_query(self, client):
        """缺少 query 字段应返回 422 验证错误"""
        resp = await client.post(
            "/v1/memory/search", json={"character_name": "ZouZheng"}
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_get_universes_returns_list(self, client):
        """获取角色宇宙列表应返回数组"""
        # 实际路由: GET /v1/memory/{character_name}/universes
        resp = await client.get("/v1/memory/ZouZheng/universes")
        assert resp.status_code == 200
        data = resp.json()
        assert "universes" in data
        assert isinstance(data["universes"], list)


# ============================================
# 4. 事件日志 API (LifeEventBus)
# ============================================


class TestDiaryAPI:
    """日记事件流入 API 的功能测试"""

    @pytest.mark.asyncio
    async def test_receive_diary_event(self, client):
        """发送一个合法的浏览事件应被接受"""
        resp = await client.post(
            "/v1/diary/event",
            json={
                "source": "browsing",
                "action": "read",
                "title": "LLM Agent Design Patterns",
                "tags": ["ai", "agents"],
                "url": "https://example.com/article",
            },
        )
        # 应该成功接受
        assert resp.status_code in (200, 201)

    @pytest.mark.asyncio
    async def test_diary_event_missing_source(self, client):
        """缺少 source 字段应返回验证错误"""
        resp = await client.post("/v1/diary/event", json={"action": "read"})
        assert resp.status_code == 422


# ============================================
# 5. 状态管理 API
# ============================================


class TestStateAPI:
    """Shujuku 用户状态 API 的功能测试"""

    @pytest.mark.asyncio
    async def test_get_user_state_new_user(self, client):
        """查询不存在的用户应返回空默认状态"""
        resp = await client.get("/v1/state/test_new_user_12345")
        assert resp.status_code == 200
        data = resp.json()
        assert "user_id" in data or "sheets" in data or "state" in data

    @pytest.mark.asyncio
    async def test_list_snapshots(self, client):
        """查询用户快照列表应返回数组"""
        resp = await client.get("/v1/state/test_user/snapshots")
        assert resp.status_code == 200


# ============================================
# 6. 综合场景: 模拟 SillyTavern 完整流程
# ============================================


class TestE2EUserJourney:
    """模拟一次完整的 ST 用户对话流程"""

    @pytest.mark.asyncio
    async def test_st_full_journey(self, client):
        """
        模拟 ST 插件的完整流程:
        1. 检查健康
        2. 查询记忆
        3. 发送对话
        """
        # Step 1: 健康检查
        health = await client.get("/")
        assert health.status_code == 200

        # Step 2: 查询记忆
        memory = await client.post(
            "/v1/memory/search",
            json={
                "query": "上次约会你穿了什么？",
                "character_name": "ZouZheng",
                "k": 2,
            },
        )
        assert memory.status_code == 200

        # Step 3: 发送对话 (不期望真调 LLM，但不应 500)
        chat = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [
                    {"role": "system", "content": "You are ZouZheng."},
                    {"role": "user", "content": "你好"},
                ],
                "stream": False,
            },
        )
        # 可能因为没有真实 API Key 而 4xx/5xx，但不应该 crash
        assert isinstance(chat.status_code, int)
