"""
功能测试: Streamlit 前端 UI (AppTest)
======================================
用 Streamlit 原生 AppTest 框架模拟用户操作，
不需要浏览器，直接在 pytest 里运行。
"""

import pytest


# ============================================
# CharLife 审核面板
# ============================================


class TestCharLifeReviewApp:
    """CharLife 审核面板功能测试"""

    def test_app_loads_without_error(self):
        """应用加载不应报错"""
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file("frontend/charlife_review_app.py").run()
        assert not at.exception, f"应用加载出错: {at.exception}"

    def test_empty_queue_shows_success_message(self):
        """待审核队列为空时应显示庆祝消息"""
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file("frontend/charlife_review_app.py").run()
        # 注入空队列
        at.session_state["pending_events"] = []
        at.run()
        # 应该有一个 success 消息
        assert len(at.success) > 0, "空队列时应显示绿色成功消息"
        # 消息内容包含"没有需要审核"
        success_text = at.success[0].value
        assert "没有" in success_text or "🎉" in success_text

    def test_with_events_shows_review_card(self):
        """有待审核事件时应显示审核卡片"""
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file("frontend/charlife_review_app.py").run()
        # 注入一个测试事件
        test_event = {
            "character": "邹峥",
            "timestamp": "2026-03-11T20:00:00",
            "details": {
                "source_topic": "今天读了一篇关于AI Agent的论文",
                "char_reaction": "这让我想起了那次在实验室的经历...",
                "emotion_tag": "好奇",
            },
        }
        at.session_state["pending_events"] = [test_event]
        at.run()
        # 应该有 info 消息显示队列数量
        assert len(at.info) > 0, "有事件时应显示队列信息"
        # 应该有按钮
        assert len(at.button) >= 2, "应有批准和驳回按钮"

    def test_refresh_button_exists_when_empty(self):
        """队列为空时应有刷新按钮"""
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file("frontend/charlife_review_app.py").run()
        at.session_state["pending_events"] = []
        at.run()
        # 应该有刷新按钮
        assert len(at.button) >= 1, "空队列时应有刷新按钮"


# ============================================
# 宇宙管理器
# ============================================


class TestUniverseManagerApp:
    """宇宙管理器前端功能测试"""

    def test_app_loads_without_error(self):
        """宇宙管理器应能正常加载（需要后端 API 连接）"""
        from streamlit.testing.v1 import AppTest

        try:
            at = AppTest.from_file(
                "frontend/universe_manager.py", default_timeout=10
            ).run()
            assert not at.exception, f"宇宙管理器加载出错: {at.exception}"
        except RuntimeError as e:
            if "timed out" in str(e):
                pytest.skip("宇宙管理器需要后端 API 连接，跳过离线测试")
            raise

    def test_page_has_title(self):
        """页面应有标题"""
        from streamlit.testing.v1 import AppTest

        try:
            at = AppTest.from_file(
                "frontend/universe_manager.py", default_timeout=10
            ).run()
            assert len(at.title) > 0 or len(at.header) > 0 or len(at.markdown) > 0, (
                "页面应有标题或标题栏元素"
            )
        except RuntimeError as e:
            if "timed out" in str(e):
                pytest.skip("宇宙管理器需要后端 API 连接，跳过离线测试")
            raise
