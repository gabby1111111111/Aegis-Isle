"""
Unit tests for Token Metrics module.
"""
import pytest
from src.aegis_isle.core.metrics import (
    TokenMetrics,
    TokenRecord,
    estimate_tokens,
    estimate_messages_tokens,
)


class TestTokenRecord:
    """TokenRecord dataclass tests."""

    def test_cost_calculation_nonzero(self):
        """Token record should calculate non-zero cost for known model."""
        record = TokenRecord(
            request_id="test-001",
            model="Qwen/Qwen2.5-7B-Instruct",
            prompt_tokens=1000,
            completion_tokens=200,
            total_tokens=1200,
            latency_ms=1500.0,
        )
        assert record.cost_usd > 0

    def test_total_tokens_consistent(self):
        """Total tokens should equal prompt + completion."""
        record = TokenRecord(
            request_id="test-002",
            model="Qwen/Qwen2.5-7B-Instruct",
            prompt_tokens=500,
            completion_tokens=100,
            total_tokens=600,
            latency_ms=800.0,
        )
        assert record.total_tokens == record.prompt_tokens + record.completion_tokens

    def test_default_model_pricing(self):
        """Unknown models should use default pricing without error."""
        record = TokenRecord(
            request_id="test-003",
            model="unknown-model-xyz",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            latency_ms=500.0,
        )
        assert record.cost_usd >= 0


class TestTokenMetrics:
    """TokenMetrics collector tests."""

    def test_record_accumulates_totals(self):
        """Recording tokens should update cumulative totals."""
        metrics = TokenMetrics()
        record = TokenRecord(
            request_id="r1",
            model="Qwen/Qwen2.5-7B-Instruct",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            latency_ms=1000.0,
        )
        metrics.record(record)
        assert metrics.total_prompt_tokens == 100
        assert metrics.total_completion_tokens == 50
        assert metrics.total_requests == 1

    def test_dashboard_returns_required_keys(self):
        """Dashboard output should contain all required top-level keys."""
        metrics = TokenMetrics()
        dashboard = metrics.get_dashboard()
        assert "token_usage" in dashboard
        assert "cost" in dashboard
        assert "latency" in dashboard
        assert "requests" in dashboard

    def test_percentile_with_no_data(self):
        """Percentile calculation should return 0 when no data exists."""
        metrics = TokenMetrics()
        dashboard = metrics.get_dashboard()
        # When no requests recorded, latency should be 0
        assert dashboard["latency"]["p50_ms"] == 0.0
        assert dashboard["latency"]["p99_ms"] == 0.0

    def test_csv_export_has_header(self):
        """CSV export should start with header row."""
        metrics = TokenMetrics()
        csv = metrics.export_csv()
        first_line = csv.strip().split("\n")[0]
        assert "request_id" in first_line
        assert "prompt_tokens" in first_line


class TestTokenEstimation:
    """Token estimation utility tests."""

    def test_estimate_tokens_nonzero(self):
        """Non-empty text should produce non-zero token count."""
        count = estimate_tokens("Hello, World!")
        assert count > 0

    def test_estimate_messages_tokens_with_overhead(self):
        """Message estimation should account for role overhead."""
        messages = [
            {"role": "system", "content": "You are an assistant."},
            {"role": "user", "content": "Hello"},
        ]
        single = estimate_tokens("You are an assistant.") + estimate_tokens("Hello")
        total = estimate_messages_tokens(messages)
        # Total should be more than raw text due to per-message overhead
        assert total > single
