"""
Unit tests for Session State Management (Context Injection).
"""
import pytest
from aegis_isle.core.state.context_injection import (
    inject_state_context,
    get_user_id_from_request,
)


class TestContextInjection:
    """Context injection middleware tests."""

    def test_inject_adds_system_message_when_empty(self):
        """Should add system message when messages list is empty."""
        result = inject_state_context([], state_markdown="## 背包\n| 物品 | 数量 |\n|---|---|\n| 剑 | 1 |")
        assert len(result) > 0
        roles = [m["role"] for m in result]
        assert "system" in roles

    def test_inject_preserves_existing_messages(self):
        """Existing user messages should be preserved after injection."""
        messages = [
            {"role": "user", "content": "你好！"},
        ]
        result = inject_state_context(messages, state_markdown="## 状态")
        user_msgs = [m for m in result if m["role"] == "user"]
        assert len(user_msgs) >= 1
        assert user_msgs[-1]["content"] == "你好！"

    def test_inject_with_empty_state_noop(self):
        """Empty state markdown should not add unnecessary content."""
        messages = [{"role": "user", "content": "test"}]
        result = inject_state_context(messages, state_markdown="")
        # Should still work without injecting empty state
        assert len(result) >= 1

    def test_inject_state_content_in_output(self):
        """State content should appear somewhere in the injected messages."""
        state = "## 背包\n| 物品 |\n|---|\n| 魔法书 |"
        result = inject_state_context([], state_markdown=state)
        all_content = " ".join(m.get("content", "") for m in result)
        assert "背包" in all_content or "魔法书" in all_content


class TestGetUserIdFromRequest:
    """User ID extraction tests."""

    def test_returns_default_for_empty_body(self):
        """Should return a default user ID for empty request body."""
        user_id = get_user_id_from_request({})
        assert isinstance(user_id, str)
        assert len(user_id) > 0

    def test_extracts_user_id_from_body(self):
        """Should extract user_id if present in request body."""
        body = {"user": "gabby_test"}
        user_id = get_user_id_from_request(body)
        assert user_id == "gabby_test"
