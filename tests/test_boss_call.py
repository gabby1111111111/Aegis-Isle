import pytest
import httpx
from unittest.mock import patch, AsyncMock
from datetime import datetime, timedelta

from aegis_isle.agents.char_life import CharLifeAgent
from aegis_isle.rag.event_logger import event_bus


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def mock_agent():
    return CharLifeAgent(memory_manager=None)


@pytest.mark.asyncio
@patch("httpx.AsyncClient.post", new_callable=AsyncMock)
async def test_boss_call_high_tension(mock_post, mock_agent):
    """测试条件1：高压情绪必须触发主动来电 (ST + ntfy双通道)"""
    mock_post.return_value = httpx.Response(200)

    reaction = {"emotion_tag": "极度狂躁", "char_reaction": "狠狠砸了一下桌子"}

    await mock_agent.evaluate_and_trigger_call("test_universe", "ZouZheng", reaction)

    # 因为有 ST 和 ntfy 两次 post 请求
    assert mock_post.call_count == 2


@pytest.mark.asyncio
@patch("httpx.AsyncClient.post", new_callable=AsyncMock)
@patch.object(event_bus, "get_last_interaction_time", new_callable=AsyncMock)
async def test_boss_call_lonely_night(mock_get_time, mock_post, mock_agent):
    """测试条件2：深夜孤独且超过12小时拉黑必须触发延时来电"""
    mock_post.return_value = httpx.Response(200)

    # 模拟距离上次互动过去了 14 小时
    mock_get_time.return_value = datetime.now() - timedelta(hours=14)

    reaction = {"emotion_tag": "孤独", "char_reaction": "看着窗外的雨"}

    # mock 时间让它觉得现在是半夜 23 点
    with patch("aegis_isle.agents.char_life.datetime") as mock_datetime:
        mock_now = datetime.now().replace(hour=23, minute=30)
        mock_datetime.now.return_value = mock_now
        mock_datetime.fromisoformat = datetime.fromisoformat

        await mock_agent.evaluate_and_trigger_call(
            "test_universe", "ZouZheng", reaction
        )

    assert mock_post.call_count == 2


@pytest.mark.asyncio
@patch("httpx.AsyncClient.post", new_callable=AsyncMock)
async def test_boss_call_no_trigger(mock_post, mock_agent):
    """测试普通情绪不会触发误报"""
    reaction = {"emotion_tag": "平静", "char_reaction": "继续看报纸"}
    await mock_agent.evaluate_and_trigger_call("test_universe", "ZouZheng", reaction)
    assert mock_post.call_count == 0


@pytest.mark.asyncio
async def test_boss_call_fallback_error(mock_agent):
    """测试若 ST 未开启，ConnectError 被优雅捕获并且投递到 EventBus 的情况"""
    reaction = {"emotion_tag": "极度压抑的烦躁", "char_reaction": "测试失败"}

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        # Mocking ST failure (ConnectError) and ntfy success
        def post_side_effect(url, **kwargs):
            if "ntfy.sh" in url:
                return httpx.Response(200)
            raise httpx.ConnectError("Connection refused")

        mock_post.side_effect = post_side_effect

        with patch.object(
            event_bus, "log_character_activity", new_callable=AsyncMock
        ) as mock_log:
            await mock_agent.evaluate_and_trigger_call(
                "test_universe", "ZouZheng", reaction
            )

            # 确认写回了未接通记录
            mock_log.assert_called_once()
            args, kwargs = mock_log.call_args
            assert kwargs["action_type"] == "missed_call_attempt"


@pytest.mark.asyncio
async def test_call_now_endpoint():
    """测试 Gabby 专用的 API 反向摇人接口"""
    from fastapi import FastAPI
    from aegis_isle.api.routers.agents import agents_router

    from fastapi.testclient import TestClient

    test_app = FastAPI()
    test_app.include_router(agents_router, prefix="/v1/agents")
    client = TestClient(test_app)

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = httpx.Response(200)

        response = client.post(
            "/v1/agents/call_now",
            json={"character_name": "ZouZheng", "universe_id": "test_universe"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert mock_post.call_count == 2
